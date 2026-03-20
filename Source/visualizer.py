"""
Visualisation and Image-Saving Utilities
=========================================
Draws bounding boxes, track IDs, FPS counter, mode overlay,
and timestamps onto frames.  Also handles async high-confidence image saving.
"""

import cv2
import numpy as np
import os
from datetime import datetime
from typing import List, Tuple, Optional
from threading import Thread, Lock
from queue import Queue


class Visualizer:
    """Renders all on-screen overlays for the detection display window."""

    def __init__(
        self,
        color_anomaly:   Tuple[int, int, int] = (0, 255, 0),
        color_pothole:   Tuple[int, int, int] = (0, 0, 255),
        show_fps:        bool = True,
        show_timestamp:  bool = True,
    ) -> None:
        self.color_anomaly  = color_anomaly    # Green
        self.color_pothole  = color_pothole    # Red
        self.show_fps       = show_fps
        self.show_timestamp = show_timestamp

        self._fps_buf:      list = []
        self._fps_buf_size: int  = 30

    # ── Track rendering (primary method for tracked objects) ─────────────

    def draw_tracks(
        self,
        frame:          np.ndarray,
        general_tracks: list,   # List[tracker.Track]
        pothole_tracks: list,   # List[tracker.Track]
    ) -> np.ndarray:

        for track in general_tracks:
            self._draw_track_box(frame, track, self.color_anomaly)
        for track in pothole_tracks:
            self._draw_track_box(frame, track, self.color_pothole)
        return frame

    def _draw_track_box(
        self,
        frame: np.ndarray,
        track,               # tracker.Track
        color: Tuple[int, int, int],
    ) -> None:
        """Draw a single track box with label and track ID."""
        x1, y1, x2, y2 = track.bbox
        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{track.label} #{track.track_id}  {track.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

        # Label background
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame, label, (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2,
        )

    # ── Legacy method — kept for backward compatibility ───────────────────

    def draw_detections(
        self,
        frame:      np.ndarray,
        boxes:      List[Tuple[int, int, int, int]],
        scores:     List[float],
        model_name: str,
    ) -> np.ndarray:
        """
        Draw raw detections (no track IDs).
        Kept for compatibility with test_system.py and other callers.
        """
        color = self.color_pothole if "Pothole" in model_name else self.color_anomaly

        for (x1, y1, x2, y2), score in zip(boxes, scores):
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{model_name}: {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(
                frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2,
            )
        return frame

    # ── Mode overlay ──────────────────────────────────────────────────────

    def add_mode_overlay(
        self,
        frame:      np.ndarray,
        mode_label: str,          # "PERFORMANCE" | "EFFICIENCY"
    ) -> np.ndarray:
        """
        Display the current model-pair mode in the bottom-right corner.

        PERFORMANCE is shown in cyan; EFFICIENCY in yellow so the operator
        can glance at the display and see the system health immediately.
        """
        color = (255, 255, 0) if mode_label == "PERFORMANCE" else (0, 255, 255)
        text  = f"Mode: {mode_label}"
        h, w  = frame.shape[:2]

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        x = w - tw - 15
        y = h - 15

        cv2.rectangle(frame, (x - 5, y - th - 8), (w - 5, y + 5), (0, 0, 0), -1)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return frame

    # ── FPS overlay ───────────────────────────────────────────────────────

    def add_fps_overlay(self, frame: np.ndarray, fps: float) -> np.ndarray:
        if not self.show_fps:
            return frame

        self._fps_buf.append(fps)
        if len(self._fps_buf) > self._fps_buf_size:
            self._fps_buf.pop(0)
        avg_fps = sum(self._fps_buf) / len(self._fps_buf)

        text = f"FPS: {avg_fps:.1f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (10, 10), (20 + tw, 20 + th), (0, 0, 0), -1)
        cv2.putText(
            frame, text, (15, 15 + th),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )
        return frame

    # ── Info overlay ──────────────────────────────────────────────────────

    def add_info_overlay(
        self,
        frame:          np.ndarray,
        frame_id:       int,
        num_detections: int,
        model_name:     str,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        text = f"Frame: {frame_id} | Det: {num_detections} | {model_name}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (0, h - th - 20), (tw + 20, h), (0, 0, 0), -1)
        cv2.putText(
            frame, text, (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
        return frame

    # ── Timestamp overlay ─────────────────────────────────────────────────

    def add_timestamp_overlay(
        self, frame: np.ndarray, timestamp: float
    ) -> np.ndarray:
        if not self.show_timestamp:
            return frame

        dt   = datetime.fromtimestamp(timestamp)
        text = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        h, w = frame.shape[:2]

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(
            frame, (w - tw - 20, 10), (w - 10, 20 + th), (0, 0, 0), -1
        )
        cv2.putText(
            frame, text, (w - tw - 15, 15 + th),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
        )
        return frame


class DetectionSaver:
    """
    Asynchronous image saver for high-confidence detections.
    Uses a background thread so saving never blocks the display loop.
    """

    def __init__(
        self,
        output_dir:          str,
        high_conf_threshold: float = 0.75,
    ) -> None:
        self.output_dir          = output_dir
        self.high_conf_threshold = high_conf_threshold

        os.makedirs(output_dir, exist_ok=True)

        self._save_q  = Queue()
        self._lock    = Lock()
        self._running = True
        self._count   = 0

        self._thread = Thread(target=self._worker, daemon=True)
        self._thread.start()
        print(f"[DetectionSaver] Initialised (threshold: {high_conf_threshold})")

    def _worker(self) -> None:
        while self._running:
            try:
                task = self._save_q.get(timeout=0.5)
                if task is None:
                    break
                frame, ts, model_name, max_conf = task
                dt    = datetime.fromtimestamp(ts)
                fname = (
                    f"{model_name}_{dt.strftime('%Y%m%d_%H%M%S_%f')}"
                    f"_{max_conf:.2f}.jpg"
                )
                cv2.imwrite(os.path.join(self.output_dir, fname), frame)
                with self._lock:
                    self._count += 1
            except Exception:
                continue

    def save_detection(
        self,
        frame:      np.ndarray,
        boxes:      List[Tuple[int, int, int, int]],
        scores:     List[float],
        model_name: str,
        timestamp:  float,
    ) -> None:
        if not scores:
            return
        max_conf = max(scores)
        if max_conf >= self.high_conf_threshold:
            self._save_q.put((frame.copy(), timestamp, model_name, max_conf))

    def save_frame(
        self, frame: np.ndarray, model_name: str, timestamp: float, conf: float
    ) -> None:
        """Save a frame unconditionally (e.g. for manual saves)."""
        self._save_q.put((frame.copy(), timestamp, model_name, conf))

    def get_save_count(self) -> int:
        with self._lock:
            return self._count

    def stop(self) -> None:
        self._running = False
        self._save_q.put(None)
        self._thread.join(timeout=2.0)
        print(f"[DetectionSaver] Stopped ({self._count} images saved)")
