"""
Adaptive Model Pair Manager — Task 1
=====================================
Manages TWO complete inference pairs and switches between them based on
real-time EMA FPS and CPU temperature using hysteresis logic.

Model pairs:
  Pair 1 — PERFORMANCE  : FP32 unquantized  (maximum accuracy)
  Pair 2 — EFFICIENCY   : INT8 quantized    (thermal stability / low CPU)

Design rules enforced here:
  • All 2 models are preloaded at construction,b then alternated by lazy switching — zero-latency switching.
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

    def __init__(self, sys_config: SystemConfig) -> None:
        """
        Load all four models and initialise switching state.

        Args:
            sys_config: Populated SystemConfig instance.
        """
        self._cfg = sys_config
        # intra  = sys_config.INTRA_OP_NUM_THREADS
        # inter  = sys_config.INTER_OP_NUM_THREADS
        # warmup = sys_config.NUM_WARMUP_RUNS

        self._perf_general = None
        self._perf_pothole = None
        self._eff_general = None
        self._eff_pothole = None
        # ── Pair 1 — Performance (FP32) ───────────────────────────────────
        # print("[ModelManager] Loading Pair 1 — Performance (FP32)…")
        # self._perf_general = OptimizedYOLODetector(
        #     sys_config.get_anomaly_config(), intra, inter, warmup
        # )
        # self._perf_pothole = OptimizedYOLODetector(
        #     sys_config.get_pothole_config(), intra, inter, warmup
        # )

        # ── Pair 2 — Efficiency (INT8 quantized) ──────────────────────────
        # quant_general_ok = os.path.exists(sys_config.ANOMALY_MODEL_QUANT_PATH)
        # quant_pothole_ok = os.path.exists(sys_config.POTHOLE_MODEL_QUANT_PATH)

        # if quant_general_ok and quant_pothole_ok:
        #     print("[ModelManager] Loading Pair 2 — Efficiency (INT8)…")
        #     self._eff_general = OptimizedYOLODetector(
        #         sys_config.get_anomaly_quant_config(), intra, inter, warmup
        #     )
        #     self._eff_pothole = OptimizedYOLODetector(
        #         sys_config.get_pothole_quant_config(), intra, inter, warmup
        #     )
        #     self._quant_available = True
        # else:
        #     # Graceful degradation: reuse FP32 detectors for Efficiency mode.
        #     # Mode switching logic still works; it just won't save resources.
        #     print(
        #         "[ModelManager] WARNING: Quantized model(s) not found. "
        #         "Efficiency mode will reuse the FP32 pair.\n"
        #         f"  Expected: {sys_config.ANOMALY_MODEL_QUANT_PATH}\n"
        #         f"            {sys_config.POTHOLE_MODEL_QUANT_PATH}"
        #     )
        #     self._eff_general = self._perf_general
        #     self._eff_pothole = self._perf_pothole
        #     self._quant_available = False

        # ── Mode state ────────────────────────────────────────────────────
        # Default to PERFORMANCE so we always start at maximum quality.
        self.current_mode: ModelMode = ModelMode.EFFICIENCY

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
            # f"Quantized pair loaded: {self._quant_available}"
        )
        self._load_active_pair()

    # ── Public API ────────────────────────────────────────────────────────

    def _load_active_pair(self) -> None:
        self._perf_general = None
        self._perf_pothole = None
        self._eff_general = None
        self._eff_pothole = None

        intra_perf  = self._cfg.INTRA_OP_NUM_THREADS_PERF
        inter_perf  = self._cfg.INTER_OP_NUM_THREADS_PERF

        intra_eff  = self._cfg.INTRA_OP_NUM_THREADS_EFF
        inter_eff  = self._cfg.INTER_OP_NUM_THREADS_EFF
        warmup = self._cfg.NUM_WARMUP_RUNS

        if self.current_mode == ModelMode.PERFORMANCE:
            self._perf_general = OptimizedYOLODetector(self._cfg.get_anomaly_config(), intra_perf, inter_perf, warmup)
            self._perf_pothole = OptimizedYOLODetector(self._cfg.get_pothole_config(), intra_perf, inter_perf, warmup)

        else:
            quant_general_ok = os.path.exists(self._cfg.ANOMALY_MODEL_QUANT_PATH)
            quant_pothole_ok = os.path.exists(self._cfg.POTHOLE_MODEL_QUANT_PATH)
            self._eff_general = OptimizedYOLODetector(self._cfg.get_anomaly_quant_config())
            self._eff_pothole = OptimizedYOLODetector(self._cfg.get_pothole_quant_config())
            if quant_general_ok and quant_pothole_ok:
                print("[ModelManager] Loading Pair 2 — Efficiency (INT8)…")
                self._eff_general = OptimizedYOLODetector(
                self._cfg.get_anomaly_quant_config(), intra_eff, inter_eff, warmup
                )
                self._eff_pothole = OptimizedYOLODetector(
                    self._cfg.get_pothole_quant_config(), intra_eff, inter_eff, warmup
                )
            else:
                print("WARNING!!!! : NO QUANTIZED MODELS FOUND")
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

        self._ema_fps = self._alpha * fps + (1.0 - self._alpha) * self._ema_fps

    def get_temperature(self) -> Optional[float]:
 
        if not self._has_temp:
            return None
        try:
            with open(self._temp_path, "r") as f:
                return float(f.read().strip()) / 1000.0
        except Exception:
            return None

    def check_and_switch(self) -> bool:

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
                self._load_active_pair()
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
                self._load_active_pair()
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
