"""
Main Application — RPi5 Object Detection System
================================================
Integrates all four upgrade tasks:

  Task 1  Adaptive model-pair switching (handled inside InferenceProcess)
  Task 2  ByteTrack object tracking      (handled in this display loop)
  Task 3  True multiprocessing pipeline  (DetectionPipeline)
  Task 4  OpenCV NMS                     (already in detector.py — unchanged)

Architecture
------------
  ┌─────────────────┐   FrameData   ┌──────────────────────┐
  │  CaptureProcess │ ─────────── ▶ │   InferenceProcess   │
  │  (camera/video) │   (Queue 1)   │  ModelPairManager    │
  └─────────────────┘               │  4 ONNX models       │
                                    │  Hysteresis switch   │
                                    └──────────────────────┘
                                           │ DetectionResult
                                           │   (Queue 2)
                                    ┌──────▼──────────────────┐
                                    │  Main Process (display) │
                                    │  ByteTracker × 2        │
                                    │  Visualizer             │
                                    │  DetectionSaver         │
                                    └─────────────────────────┘
"""

import cv2
import time
import argparse
import sys

from config import config
from pipeline import DetectionPipeline
from tracker import ByteTracker
from visualizer import Visualizer, DetectionSaver


class RPi5DetectionSystem:
    """
    Top-level controller.

    All heavy lifting (model loading, inference, frame capture) runs in
    separate processes.  This class owns the display loop, the two
    ByteTrackers, and the visualiser.
    """

    def __init__(self, save_detections: bool = True) -> None:
        print("=" * 60)
        print("  RPi5 Object Detection System  (v2 — multiprocessing)")
        print("=" * 60)

        # ── Visualiser ────────────────────────────────────────────────────
        print("\n[1/4] Initialising visualiser…")
        self.visualizer = Visualizer(
            color_anomaly  = config.COLOR_ANOMALY,
            color_pothole  = config.COLOR_POTHOLE,
            show_fps       = config.DISPLAY_FPS,
            show_timestamp = config.SHOW_TIMESTAMPS,
        )

        # ── Detection saver ───────────────────────────────────────────────
        print("[2/4] Initialising detection saver…")
        self.saver: DetectionSaver | None = None
        if save_detections:
            self.saver = DetectionSaver(
                output_dir          = config.OUTPUT_DIR,
                high_conf_threshold = config.HIGH_CONF_SAVE_THRESHOLD,
            )

        # ── ByteTrackers — one per model type ─────────────────────────────
        print("[3/4] Initialising ByteTrackers…")
        self.general_tracker = ByteTracker(
            label               = "Obstacle",
            max_lost_frames     = config.TRACKER_MAX_LOST_FRAMES,
            iou_threshold       = config.TRACKER_IOU_THRESHOLD,
            high_conf_threshold = config.TRACKER_HIGH_CONF_THRESHOLD,
            low_conf_threshold  = config.TRACKER_LOW_CONF_THRESHOLD,
        )
        self.pothole_tracker = ByteTracker(
            label               = "Pothole",
            max_lost_frames     = config.TRACKER_MAX_LOST_FRAMES,
            iou_threshold       = config.TRACKER_IOU_THRESHOLD,
            high_conf_threshold = config.TRACKER_HIGH_CONF_THRESHOLD,
            low_conf_threshold  = config.TRACKER_LOW_CONF_THRESHOLD,
        )

        # ── Multiprocessing pipeline (workers started in run_*) ───────────
        print("[4/4] Pipeline ready (workers start on run).")
        self.pipeline: DetectionPipeline | None = None

        print("\n✓ System initialised — call run_camera() or run_video()\n")

    # ── Entry points ──────────────────────────────────────────────────────

    def run_camera(self, camera_id: int = 0) -> None:
        """Start live detection from the Pi camera or a USB webcam."""
        print(f"\n▶ Starting camera detection (ID: {camera_id})")
        src = "pi_camera" if camera_id == 0 else camera_id
        self._run(
            camera_src = src,
            width      = config.CAMERA_WIDTH,
            height     = config.CAMERA_HEIGHT,
            fps        = config.CAMERA_FPS,
            is_video   = False,
        )

    def run_video(self, video_path: str) -> None:
        """Start detection on a video file."""
        print(f"\n▶ Starting video detection: {video_path}")
        self._run(
            camera_src = video_path,
            width      = config.CAMERA_WIDTH,
            height     = config.CAMERA_HEIGHT,
            fps        = config.CAMERA_FPS,
            is_video   = True,
        )

    # ── Core run method ───────────────────────────────────────────────────

    def _run(
        self,
        camera_src: object,
        width:      int,
        height:     int,
        fps:        int,
        is_video:   bool,
    ) -> None:
        """
        Build the pipeline, start worker processes, then run the display loop
        in the main process (cv2.imshow must be on the main thread).
        """
        print("\nControls:")
        print("  'q' — Quit")
        print("  'p' — Pause / Resume")
        print("  's' — Save current frame manually")
        print("-" * 60)

        self.pipeline = DetectionPipeline(
            max_queue_size = config.FRAME_QUEUE_SIZE,
            max_latency_ms = config.MAX_LATENCY_MS,
        )
        self.pipeline.start(
            camera_src = camera_src,
            width      = width,
            height     = height,
            fps        = fps,
        )

        try:
            self._display_loop(is_video)
        except KeyboardInterrupt:
            print("\n⚠ Interrupted by user")
        finally:
            self._cleanup()

    def _display_loop(self, is_video: bool) -> None:
        """
        Main thread loop: fetch results, run trackers, render, display.

        This is the ONLY place cv2.imshow is called (main-thread requirement).
        The two ByteTrackers live here:
          - general_tracker updated when is_pothole_frame == False
          - pothole_tracker updated when is_pothole_frame == True
        """
        paused         = False
        frame_count    = 0
        last_result    = None

        # FPS measurement
        fps_start      = time.time()
        fps_frames     = 0
        current_fps    = 0.0

        # Track lists (persisted across frames for combined display)
        general_tracks: list = []
        pothole_tracks: list = []

        while True:
            # ── Check for end-of-stream / process death ────────────────────
            if not self.pipeline.is_alive():
                # Workers may have finished (e.g. end of video)
                # Drain any remaining results before exiting
                remaining = self.pipeline.get_result(timeout=0.2)
                if remaining is None:
                    print("\n✓ Pipeline finished.")
                    break
                # Process this last result
                result = remaining
            else:
                result = self.pipeline.get_result(timeout=0.05)

            if result is None:
                # No new result yet — redisplay the last frame with updated FPS
                if last_result is not None and not paused:
                    self._show_frame(
                        last_result, general_tracks, pothole_tracks,
                        current_fps, frame_count,
                    )
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\n✓ Stopped by user.")
                    break
                elif key == ord("p"):
                    paused = not paused
                    print(f"  {'⏸ Paused' if paused else '▶ Resumed'}")
                continue

            if paused:
                # Still consume from queue while paused to avoid backlog
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("p"):
                    paused = False
                    print("  ▶ Resumed")
                continue

            # ── Update the appropriate ByteTracker ─────────────────────────
            #
            # The inference process alternates models each frame:
            #   is_pothole_frame = True  → pothole model ran
            #   is_pothole_frame = False → general model ran
            #
            # We ONLY call update() on the relevant tracker to avoid
            # marking the other tracker's objects as lost prematurely.
            if result.is_pothole_frame:
                pothole_tracks = self.pothole_tracker.update(
                    result.boxes, result.scores
                )
            else:
                general_tracks = self.general_tracker.update(
                    result.boxes, result.scores
                )

            last_result  = result
            frame_count += 1

            # ── Render and display ─────────────────────────────────────────
            display_frame = self._show_frame(
                result, general_tracks, pothole_tracks,
                current_fps, frame_count,
            )

            # ── Save high-confidence detections ───────────────────────────
            if self.saver and result.scores:
                self.saver.save_detection(
                    display_frame,
                    result.boxes,
                    result.scores,
                    result.model_name,
                    result.timestamp,
                )

            # ── FPS calculation ────────────────────────────────────────────
            fps_frames += 1
            elapsed     = time.time() - fps_start
            if elapsed >= 1.0:
                current_fps = fps_frames / elapsed
                fps_frames  = 0
                fps_start   = time.time()
                self._print_stats(
                    current_fps, frame_count,
                    len(general_tracks) + len(pothole_tracks),
                    result.mode_label,
                )

            # ── Key handling ──────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n✓ Stopped by user.")
                break
            elif key == ord("p"):
                paused = not paused
                print(f"  {'⏸ Paused' if paused else '▶ Resumed'}")
            elif key == ord("s") and display_frame is not None:
                path = f"{config.OUTPUT_DIR}/manual_{frame_count}.jpg"
                cv2.imwrite(path, display_frame)
                print(f"  💾 Saved: {path}")

    # ── Rendering helper ──────────────────────────────────────────────────

    def _show_frame(
        self,
        result:          object,      # DetectionResult
        general_tracks:  list,
        pothole_tracks:  list,
        fps:             float,
        frame_count:     int,
    ):
        """Composite all overlays onto the frame and push to the window."""
        frame = result.frame.copy()

        # Draw tracked objects with persistent IDs
        frame = self.visualizer.draw_tracks(frame, general_tracks, pothole_tracks)

        # Overlays
        frame = self.visualizer.add_fps_overlay(frame, fps)
        frame = self.visualizer.add_timestamp_overlay(frame, result.timestamp)
        frame = self.visualizer.add_mode_overlay(frame, result.mode_label)
        frame = self.visualizer.add_info_overlay(
            frame,
            frame_count,
            len(general_tracks) + len(pothole_tracks),
            result.model_name,
        )

        cv2.imshow("RPi5 Object Detection", frame)
        return frame

    # ── Logging ───────────────────────────────────────────────────────────

    @staticmethod
    def _print_stats(fps, frame_count, n_tracks, mode):
        save_info = ""
        print(
            f"FPS: {fps:.1f} | Frame: {frame_count} | "
            f"Tracked: {n_tracks} | Mode: {mode}{save_info}"
        )

    # ── Cleanup ───────────────────────────────────────────────────────────

    def _cleanup(self) -> None:
        print("\n" + "=" * 60)
        print("Cleaning up…")

        if self.pipeline:
            self.pipeline.stop()

        if self.saver:
            self.saver.stop()

        cv2.destroyAllWindows()

        print("✓ Cleanup complete.")
        print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RPi5 Object Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Live camera (default: Pi CSI camera)
  python main.py

  # Specific camera index
  python main.py --source camera --camera-id 1

  # Video file
  python main.py --source road_video.mp4

  # Disable auto-save
  python main.py --no-save

  # Override confidence threshold
  python main.py --conf-threshold 0.65
        """,
    )

    parser.add_argument(
        "--source", type=str, default="camera",
        help='Input: "camera" or path to a video file (default: camera)',
    )
    parser.add_argument(
        "--camera-id", type=int, default=0,
        help="Camera index when --source camera (default: 0)",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Disable automatic saving of high-confidence detections",
    )
    parser.add_argument(
        "--conf-threshold", type=float, default=None,
        help="Override confidence threshold for both models",
    )
    parser.add_argument(
        "--save-threshold", type=float, default=None,
        help="Override save threshold for high-confidence detections",
    )

    args = parser.parse_args()

    # Apply config overrides
    if args.conf_threshold is not None:
        config.ANOMALY_CONF_THRESHOLD = args.conf_threshold
        config.POTHOLE_CONF_THRESHOLD = args.conf_threshold

    if args.save_threshold is not None:
        config.HIGH_CONF_SAVE_THRESHOLD = args.save_threshold

    # Initialise system
    try:
        system = RPi5DetectionSystem(save_detections=not args.no_save)
    except Exception as exc:
        print(f"✗ Error initialising system: {exc}")
        sys.exit(1)

    # Run
    try:
        if args.source.lower() == "camera":
            system.run_camera(camera_id=args.camera_id)
        else:
            system.run_video(video_path=args.source)
    except Exception as exc:
        print(f"✗ Runtime error: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Required for multiprocessing on all platforms
    mp_ctx = __import__("multiprocessing")
    mp_ctx.set_start_method("fork", force=True)   # Linux default; explicit for clarity
    main()
