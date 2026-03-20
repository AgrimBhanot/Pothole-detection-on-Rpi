"""
True Multiprocessing Pipeline — Task 3
=======================================
Separates the three compute stages into independent OS processes:

  Process 1 — Camera / Capture
      Reads frames from Pi camera or video file.
      Drops frames if the capture queue is full (never blocks).

  Process 2 — Inference
      Creates ModelPairManager (all 4 ONNX models) on startup.
      Alternates between general and pothole detectors each frame.
      Monitors FPS + temperature; switches model pair via hysteresis.
      Drops stale frames (latency > MAX_LATENCY_MS).

  Process 3 (Main thread) — Tracking + Display
      Reads DetectionResult objects from the detection queue.
      Runs ByteTrackers and renders the frame.
      (The main thread serves as "Process 3" since cv2.imshow requires it.)

Queue design (zero-latency):
    maxsize=1  → putting a new item when full discards the old one
                 so the inference / display always sees the latest frame.
"""

import multiprocessing as mp
import time
import queue as _queue_module
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List


# ---------------------------------------------------------------------------
# Data containers (must be picklable — plain dataclasses are fine)
# ---------------------------------------------------------------------------

@dataclass
class FrameData:
    """Minimal frame envelope passed from capture → inference."""
    frame:      object    # np.ndarray (picklable via pickle/shared memory)
    frame_id:   int
    timestamp:  float     # time.time() at capture


@dataclass
class DetectionResult:
    """Detection output passed from inference → display."""
    frame:            object    # np.ndarray — the source frame
    boxes:            list      # List[Tuple[int,int,int,int]]
    scores:           list      # List[float]
    model_name:       str
    timestamp:        float
    frame_id:         int
    is_pothole_frame: bool      # True if pothole model ran; False if general
    mode_label:       str       # "PERFORMANCE" | "EFFICIENCY"


# ------------------------------------
# Module-level worker functions  (picklable for multiprocessing)
# -------------------------------------------------------------------

def _capture_worker(
    camera_src: object,          # "pi_camera" | int | str path
    width:      int,
    height:     int,
    fps:        int,
    capture_q:  mp.Queue,
    running:    mp.Value,        # mp.Value('i')
) -> None:

    # Import inside worker — safe for both fork and spawn start methods
    from camera import ThreadedCamera

    try:
        cam = ThreadedCamera(
            src=camera_src,
            width=width,
            height=height,
            fps=fps,
        )
    except Exception as exc:
        print(f"[CaptureWorker] Failed to open camera: {exc}")
        _put_sentinel(capture_q)
        return

    frame_id = 0
    print("[CaptureWorker] Started.")

    while running.value:
        ret, frame = cam.read()
        if not ret or frame is None:
            # End of video file or camera disconnected
            break
        frame_data = FrameData(
            frame=frame,
            frame_id=frame_id,
            timestamp=time.time(),
        )
        frame_id += 1
        _put_drop_old(capture_q, frame_data)

    # Signal downstream that no more frames are coming
    _put_sentinel(capture_q)
    cam.release()
    print("[CaptureWorker] Stopped.")


def _inference_worker(
    capture_q:      mp.Queue,
    detection_q:    mp.Queue,
    running:        mp.Value,    # mp.Value('i')
    mode_val:       mp.Value,    # mp.Value('i')  1=PERF, 0=EFF
    max_latency_ms: int,
) -> None:

    from config import config
    from model_manager import ModelPairManager, ModelMode

    print("[InferenceWorker] Initialising model pair manager…")
    try:
        manager = ModelPairManager(config)
    except Exception as exc:
        print(f"[InferenceWorker] FATAL: Could not load models: {exc}")
        _put_sentinel(detection_q)
        return

    use_pothole_frame   = False        # Alternating flag
    frame_times: deque  = deque(maxlen=30)
    last_switch_check   = time.monotonic()

    print("[InferenceWorker] Started.")

    while running.value:
        # --- Grab the next frame (drop stale items) -----------------------
        try:
            frame_data = capture_q.get(timeout=0.1)
        except _queue_module.Empty:
            continue

        if frame_data is None:          # End-of-stream sentinel
            _put_sentinel(detection_q)
            break

        # --- Latency guard — discard frames that are too old --------------
        latency_ms = (time.time() - frame_data.timestamp) * 1000.0
        if max_latency_ms > 0 and latency_ms > max_latency_ms:
            continue                    # Frame is stale; skip it

        # --- Select active model pair (zero latency) ----------------------
        general_det, pothole_det = manager.get_active_detectors()

        # --- Alternate between models each frame --------------------------
        t0 = time.time()
        if use_pothole_frame:
            boxes, scores = pothole_det.detect(frame_data.frame)
            model_name    = pothole_det.get_name()
            is_pothole    = True
        else:
            boxes, scores = general_det.detect(frame_data.frame)
            model_name    = general_det.get_name()
            is_pothole    = False
        t1 = time.time()

        use_pothole_frame = not use_pothole_frame   # Flip for next frame

        # --- Update EMA FPS -----------------------------------------------
        inference_s = t1 - t0
        if inference_s > 0:
            frame_times.append(inference_s)
            if len(frame_times) >= 5:
                avg_t = sum(frame_times) / len(frame_times)
                manager.update_fps(1.0 / avg_t)

        # --- Periodic hysteresis check (once per second) ------------------
        if t1 - last_switch_check >= 1.0:
            manager.check_and_switch()
            # Publish current mode to the shared integer so the main
            # process can display it without inter-process function calls.
            mode_val.value = (
                1 if manager.current_mode == ModelMode.PERFORMANCE else 0
            )
            last_switch_check = t1

        # --- Build result and push to display queue -----------------------
        result = DetectionResult(
            frame            = frame_data.frame,
            boxes            = boxes,
            scores           = scores,
            model_name       = model_name,
            timestamp        = frame_data.timestamp,
            frame_id         = frame_data.frame_id,
            is_pothole_frame = is_pothole,
            mode_label       = manager.mode_label,
        )
        _put_drop_old(detection_q, result)

    print("[InferenceWorker] Stopped.")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _put_drop_old(q: mp.Queue, item) -> None:
    """
    Non-blocking put.  If the queue is full, discard the oldest item and
    insert the new one.  This guarantees the consumer always sees the
    LATEST data — never a backlog.
    """
    try:
        q.put(item, block=False)
    except _queue_module.Full:
        try:
            q.get_nowait()          # Discard stale item
            q.put(item, block=False)
        except Exception:
            pass                    # Race condition — just skip this frame


def _put_sentinel(q: mp.Queue) -> None:
    """Put a ``None`` sentinel to signal end-of-stream to downstream workers."""
    try:
        q.put(None, timeout=2.0)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Pipeline controller
# ---------------------------------------------------------------------------

class DetectionPipeline:


    def __init__(
        self,
        max_queue_size: int = 1,
        max_latency_ms: int = 200,
    ) -> None:
        """
        Args:
            max_queue_size: Queue depth.  Keep at 1 for zero-latency behaviour.
            max_latency_ms: Discard inference frames older than this value.
        """
        self.max_latency_ms = max_latency_ms

        # Inter-process queues  (maxsize=1 → drop-old on overflow)
        self.capture_q   = mp.Queue(maxsize=max_queue_size)
        self.detection_q = mp.Queue(maxsize=max_queue_size)

        # Shared state visible from both processes and the main thread
        self.running   = mp.Value("i", 1)   # 1 = running
        self.mode_val  = mp.Value("i", 1)   # 1 = PERFORMANCE, 0 = EFFICIENCY

        self._cap_proc: Optional[mp.Process] = None
        self._inf_proc: Optional[mp.Process] = None

        print("[Pipeline] Initialised.")

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(
        self,
        camera_src: object,   # "pi_camera" | int | str path
        width:      int   = 640,
        height:     int   = 480,
        fps:        int   = 30,
    ) -> None:
        """
        Spawn the camera capture and inference processes.

        The camera is initialised INSIDE the capture process so that
        picamera2 is never shared across a fork boundary.
        """
        self.running.value = 1

        # --- Capture process ----------------------------------------------
        self._cap_proc = mp.Process(
            target  = _capture_worker,
            args    = (camera_src, width, height, fps,
                       self.capture_q, self.running),
            daemon  = True,
            name    = "CaptureProcess",
        )
        self._cap_proc.start()

        # --- Inference process --------------------------------------------
        self._inf_proc = mp.Process(
            target  = _inference_worker,
            args    = (self.capture_q, self.detection_q,
                       self.running, self.mode_val, self.max_latency_ms),
            daemon  = True,
            name    = "InferenceProcess",
        )
        self._inf_proc.start()

        print(
            f"[Pipeline] Started — "
            f"CaptureProcess PID {self._cap_proc.pid}, "
            f"InferenceProcess PID {self._inf_proc.pid}"
        )

    def stop(self) -> None:
        """Signal all workers to stop and drain queues."""
        self.running.value = 0

        # Drain queues so worker joins don't block on full queue
        for q in (self.capture_q, self.detection_q):
            _drain(q)

        if self._cap_proc and self._cap_proc.is_alive():
            self._cap_proc.join(timeout=3.0)
        if self._inf_proc and self._inf_proc.is_alive():
            self._inf_proc.join(timeout=5.0)   # Inference may need time to finish

        print("[Pipeline] Stopped.")

    # ── Data access ────────────────────────────────────────────────────────

    def get_result(self, timeout: float = 0.05) -> Optional[DetectionResult]:

        try:
            item = self.detection_q.get(timeout=timeout)
            # ``None`` is the end-of-stream sentinel
            return item   # May be None for end-of-stream; caller handles it
        except _queue_module.Empty:
            return None

    def get_current_mode(self) -> str:
        """Human-readable mode label read from shared memory."""
        return "PERFORMANCE" if self.mode_val.value == 1 else "EFFICIENCY"

    def get_queue_sizes(self) -> Tuple[int, int]:
        """Return (capture_queue_size, detection_queue_size)."""
        return self.capture_q.qsize(), self.detection_q.qsize()

    def is_alive(self) -> bool:
        """True while both worker processes are still running."""
        cap_ok = self._cap_proc is not None and self._cap_proc.is_alive()
        inf_ok = self._inf_proc is not None and self._inf_proc.is_alive()
        return cap_ok and inf_ok


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------

def _drain(q: mp.Queue) -> None:
    """Empty a queue without blocking."""
    while True:
        try:
            q.get_nowait()
        except Exception:
            break
