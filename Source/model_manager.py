"""
Adaptive Model Pair Manager — Task 1
=====================================
Manages TWO complete inference pairs and switches between them based on
real-time EMA FPS and CPU temperature using hysteresis logic.

Model pairs:
  Pair 1 — PERFORMANCE  : FP32 unquantized  (maximum accuracy)
  Pair 2 — EFFICIENCY   : INT8 quantized    (thermal stability / low CPU)

Design rules enforced here:
  • All 4 models are preloaded at construction — zero-latency switching.
  • Active pair is selected with a single conditional (O(1), no I/O).
  • EMA smoothing prevents reacting to single-frame FPS spikes.
  • Separate downgrade / upgrade thresholds create a hysteresis dead-band.
  • A cooldown period prevents rapid oscillation between modes.
  • Switching state is logged for debugging.
"""

import os
import time
from enum import Enum
from typing import Tuple, Optional

from config import SystemConfig
from detector import OptimizedYOLODetector


class ModelMode(Enum):
    """Active inference pair."""
    PERFORMANCE = "PERFORMANCE"   # FP32 — use when resources allow
    EFFICIENCY  = "EFFICIENCY"    # INT8 — use under thermal / FPS pressure


class ModelPairManager:
    """
    Pre-loads and manages all four ONNX detectors.

    The manager is always instantiated INSIDE the inference subprocess so that
    ONNX Runtime session objects are never pickled across process boundaries.

    Switching is zero-latency: ``get_active_detectors()`` just reads a
    Python attribute — no file I/O, no session re-creation.
    """

    def __init__(self, sys_config: SystemConfig) -> None:
        """
        Load all four models and initialise switching state.

        Args:
            sys_config: Populated SystemConfig instance.
        """
        self._cfg = sys_config
        intra  = sys_config.INTRA_OP_NUM_THREADS
        inter  = sys_config.INTER_OP_NUM_THREADS
        warmup = sys_config.NUM_WARMUP_RUNS

        # ── Pair 1 — Performance (FP32) ───────────────────────────────────
        print("[ModelManager] Loading Pair 1 — Performance (FP32)…")
        self._perf_general = OptimizedYOLODetector(
            sys_config.get_anomaly_config(), intra, inter, warmup
        )
        self._perf_pothole = OptimizedYOLODetector(
            sys_config.get_pothole_config(), intra, inter, warmup
        )

        # ── Pair 2 — Efficiency (INT8 quantized) ──────────────────────────
        quant_general_ok = os.path.exists(sys_config.ANOMALY_MODEL_QUANT_PATH)
        quant_pothole_ok = os.path.exists(sys_config.POTHOLE_MODEL_QUANT_PATH)

        if quant_general_ok and quant_pothole_ok:
            print("[ModelManager] Loading Pair 2 — Efficiency (INT8)…")
            self._eff_general = OptimizedYOLODetector(
                sys_config.get_anomaly_quant_config(), intra, inter, warmup
            )
            self._eff_pothole = OptimizedYOLODetector(
                sys_config.get_pothole_quant_config(), intra, inter, warmup
            )
            self._quant_available = True
        else:
            # Graceful degradation: reuse FP32 detectors for Efficiency mode.
            # Mode switching logic still works; it just won't save resources.
            print(
                "[ModelManager] WARNING: Quantized model(s) not found. "
                "Efficiency mode will reuse the FP32 pair.\n"
                f"  Expected: {sys_config.ANOMALY_MODEL_QUANT_PATH}\n"
                f"            {sys_config.POTHOLE_MODEL_QUANT_PATH}"
            )
            self._eff_general = self._perf_general
            self._eff_pothole = self._perf_pothole
            self._quant_available = False

        # ── Mode state ────────────────────────────────────────────────────
        # Default to PERFORMANCE so we always start at maximum quality.
        self.current_mode: ModelMode = ModelMode.PERFORMANCE

        # ── EMA FPS smoother ──────────────────────────────────────────────
        # Initialised at a neutral mid-range value so we don't immediately
        # trigger a mode switch on the first few frames.
        self._ema_fps: float = 15.0
        self._alpha:   float = sys_config.FPS_EMA_ALPHA   # e.g. 0.2

        # ── Thermal sensor ────────────────────────────────────────────────
        self._temp_path = "/sys/class/thermal/thermal_zone0/temp"
        self._has_temp  = os.path.exists(self._temp_path)

        # ── Hysteresis cooldown ───────────────────────────────────────────
        self._last_switch: float = 0.0   # monotonic time of last switch
        self._cooldown:    float = sys_config.SWITCH_COOLDOWN_SECONDS

        print(
            f"[ModelManager] Ready. Default mode: {self.current_mode.value}  |  "
            f"Quantized pair loaded: {self._quant_available}"
        )

    # ── Public API ────────────────────────────────────────────────────────

    def get_active_detectors(
        self,
    ) -> Tuple[OptimizedYOLODetector, OptimizedYOLODetector]:
        """
        Return ``(general_detector, pothole_detector)`` for the active mode.

        This is the hot path — called once per inference frame.
        It is a pure in-memory attribute lookup with no side effects.
        """
        if self.current_mode == ModelMode.PERFORMANCE:
            return self._perf_general, self._perf_pothole
        return self._eff_general, self._eff_pothole

    def update_fps(self, fps: float) -> None:
        """
        Feed a new per-frame FPS sample into the exponential moving average.

        Should be called once per completed inference cycle.
        The EMA smooths out short-lived spikes so we don't trigger a mode
        switch based on a single slow frame.
        """
        self._ema_fps = self._alpha * fps + (1.0 - self._alpha) * self._ema_fps

    def get_temperature(self) -> Optional[float]:
        """
        Read CPU temperature in Celsius via the sysfs thermal interface.

        Returns ``None`` if the sensor is unavailable (e.g. non-RPi system).
        """
        if not self._has_temp:
            return None
        try:
            with open(self._temp_path, "r") as f:
                return float(f.read().strip()) / 1000.0
        except Exception:
            return None

    def check_and_switch(self) -> bool:
        """
        Evaluate hysteresis switching rules and change mode if warranted.

        Switching rules
        ---------------
        ↓  PERFORMANCE → EFFICIENCY  (downgrade)  when ANY of:
              EMA FPS  < FPS_DOWNGRADE_THRESHOLD  (default 8 fps)
              CPU temp > TEMP_DOWNGRADE_THRESHOLD (default 75 °C)

        ↑  EFFICIENCY → PERFORMANCE  (upgrade)    when ALL of:
              EMA FPS  > FPS_UPGRADE_THRESHOLD    (default 12 fps)
              CPU temp < TEMP_UPGRADE_THRESHOLD   (default 65 °C)

        The gap between downgrade (8 fps / 75 °C) and upgrade (12 fps / 65 °C)
        is the hysteresis dead-band. A cooldown period additionally limits
        switch frequency.

        Returns:
            ``True`` if a mode switch occurred this call.
        """
        now = time.monotonic()

        # Enforce cooldown — prevents oscillation even when near thresholds
        if now - self._last_switch < self._cooldown:
            return False

        cfg  = self._cfg
        fps  = self._ema_fps
        temp = self.get_temperature()

        if self.current_mode == ModelMode.PERFORMANCE:
            # --- Downgrade check -------------------------------------------
            low_fps  = fps < cfg.FPS_DOWNGRADE_THRESHOLD
            overtemp = (temp is not None) and (temp > cfg.TEMP_DOWNGRADE_THRESHOLD)

            if low_fps or overtemp:
                reasons: list = []
                if low_fps:
                    reasons.append(
                        f"low FPS ({fps:.1f} < {cfg.FPS_DOWNGRADE_THRESHOLD})"
                    )
                if overtemp:
                    reasons.append(
                        f"thermal spike ({temp:.1f}°C > {cfg.TEMP_DOWNGRADE_THRESHOLD}°C)"
                    )
                self.current_mode = ModelMode.EFFICIENCY
                self._last_switch = now
                print(
                    f"[ModelManager] Switched to EFFICIENCY due to: "
                    f"{', '.join(reasons)}"
                )
                return True

        else:  # ModelMode.EFFICIENCY
            # --- Upgrade check --------------------------------------------
            good_fps  = fps > cfg.FPS_UPGRADE_THRESHOLD
            # If no thermal sensor, assume temperature is fine
            cool_temp = (temp is None) or (temp < cfg.TEMP_UPGRADE_THRESHOLD)

            if good_fps and cool_temp:
                self.current_mode = ModelMode.PERFORMANCE
                self._last_switch = now
                temp_str = f"{temp:.1f}°C" if temp is not None else "N/A"
                print(
                    f"[ModelManager] Restored PERFORMANCE after recovery  "
                    f"(EMA FPS: {fps:.1f}, Temp: {temp_str})"
                )
                return True

        return False

    # ── Properties for monitoring ─────────────────────────────────────────

    @property
    def ema_fps(self) -> float:
        """Current smoothed FPS estimate used for switching decisions."""
        return self._ema_fps

    @property
    def mode_label(self) -> str:
        """Human-readable label for the current mode (for UI overlays)."""
        return self.current_mode.value
