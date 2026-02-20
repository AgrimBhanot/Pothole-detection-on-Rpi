"""
Main application for RPi5 Object Detection System
Optimized for high performance with dual models and multiprocessing
"""

import cv2
import time
import argparse
import sys
from typing import Optional

from config import config
from detector import OptimizedYOLODetector
from camera import ThreadedCamera, VideoCapture
from pipeline import DetectionPipeline
from visualizer import Visualizer, DetectionSaver


class RPi5DetectionSystem:
    """
    Main detection system for Raspberry Pi 5
    Integrates all components for high-performance object detection
    """
    
    def __init__(self, use_dual_models: bool = True, save_detections: bool = True):
        """
        Initialize the detection system
        
        Args:
            use_dual_models: Whether to use alternating dual models
            save_detections: Whether to save high-confidence detections
        """
        print("=" * 60)
        print("RPi5 Object Detection System")
        print("=" * 60)
        
        self.use_dual_models = use_dual_models
        self.save_detections = save_detections
        
        # Initialize models
        print("\n[1/5] Loading models...")
        self.detector_anomaly = OptimizedYOLODetector(
            model_config=config.get_anomaly_config(),
            intra_threads=config.INTRA_OP_NUM_THREADS,
            inter_threads=config.INTER_OP_NUM_THREADS,
            warmup_runs=config.NUM_WARMUP_RUNS
        )
        
        if use_dual_models:
            self.detector_pothole = OptimizedYOLODetector(
                model_config=config.get_pothole_config(),
                intra_threads=config.INTRA_OP_NUM_THREADS,
                inter_threads=config.INTER_OP_NUM_THREADS,
                warmup_runs=config.NUM_WARMUP_RUNS
            )
        else:
            self.detector_pothole = None
        
        # Initialize visualizer
        print("\n[2/5] Initializing visualizer...")
        self.visualizer = Visualizer(
            color_anomaly=config.COLOR_ANOMALY,
            color_pothole=config.COLOR_POTHOLE,
            show_fps=config.DISPLAY_FPS,
            show_timestamp=config.SHOW_TIMESTAMPS
        )
        
        # Initialize saver
        print("\n[3/5] Initializing detection saver...")
        if save_detections:
            self.saver = DetectionSaver(
                output_dir=config.OUTPUT_DIR,
                high_conf_threshold=config.HIGH_CONF_SAVE_THRESHOLD
            )
        else:
            self.saver = None
        
        # Pipeline will be initialized when run() is called
        self.pipeline = None
        self.camera = None
        
        print("\n[4/5] System ready!")
    
    def run_camera(self, camera_id: int = 0):
        """
        Run detection on camera feed
        
        Args:
            camera_id: Camera index
        """
        print(f"\n[5/5] Starting camera detection (ID: {camera_id})...")
        print("\nControls:")
        print("  'q' - Quit")
        print("  'p' - Pause/Resume")
        print("  's' - Save current frame manually")
        print("-" * 60)
        
        # Initialize camera
        try:
            self.camera = ThreadedCamera(
                src=camera_id,
                width=config.CAMERA_WIDTH,
                height=config.CAMERA_HEIGHT,
                fps=config.CAMERA_FPS
            )
        except Exception as e:
            print(f"❌ Error: Failed to initialize camera: {e}")
            return
        
        # Initialize pipeline
        self.pipeline = DetectionPipeline(
            max_queue_size=config.FRAME_QUEUE_SIZE,
            max_latency_ms=config.MAX_LATENCY_MS
        )
        
        # Start pipeline processes
        # Note: In multiprocessing, we can't directly pass class instances
        # So we'll use a simplified single-process approach for now
        # For true multiprocessing, you'd need to use queues with serializable data
        
        self._run_detection_loop()
    
    def run_video(self, video_path: str):
        """
        Run detection on video file
        
        Args:
            video_path: Path to video file
        """
        print(f"\n[5/5] Starting video detection: {video_path}...")
        print("\nControls:")
        print("  'q' - Quit")
        print("  'p' - Pause/Resume")
        print("  's' - Save current frame manually")
        print("-" * 60)
        
        # Initialize video capture
        try:
            self.camera = VideoCapture(video_path=video_path)
        except Exception as e:
            print(f"❌ Error: Failed to load video: {e}")
            return
        
        # Initialize pipeline
        self.pipeline = DetectionPipeline(
            max_queue_size=config.FRAME_QUEUE_SIZE,
            max_latency_ms=config.MAX_LATENCY_MS
        )
        
        self._run_detection_loop()
    
    def _run_detection_loop(self, is_video: bool = False):
        """Main detection loop (simplified single-process version)
        Arg: is_video, checking if it is video
        """
        paused = False
        use_model_1 = True
        frame_count = 0
        
        # FPS calculation
        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = 0.0
        
        # Persistent boxes for smooth visualization
        last_boxes_1 = []
        last_scores_1 = []
        last_boxes_2 = []
        last_scores_2 = []
        
        try:
            while self.camera.is_opened():
                loop_start = time.time()
                
                if not paused:
                    ret, frame = self.camera.read()
                    if not ret:
                        if is_video:
                            print("\n✓ End of video reached")
                        else:
                            print("\n✓ Camera disconnected")
                        break
                    
                    frame_count += 1
                    timestamp = time.time()
                    
                    # Select detector
                    if self.use_dual_models and config.USE_ALTERNATE_MODELS:
                        detector = self.detector_anomaly if use_model_1 else self.detector_pothole
                    else:
                        detector = self.detector_anomaly
                    
                    model_name = detector.get_name()
                    
                    # Run detection
                    boxes, scores = detector.detect(frame)
                    
                    # Store results for persistence
                    if use_model_1 or not self.use_dual_models:
                        last_boxes_1 = boxes
                        last_scores_1 = scores
                        persistent_boxes = last_boxes_2
                        persistent_scores = last_scores_2
                    else:
                        last_boxes_2 = boxes
                        last_scores_2 = scores
                        persistent_boxes = last_boxes_1
                        persistent_scores = last_scores_1
                    
                    # Combine current and persistent detections
                    combined_boxes = boxes + persistent_boxes
                    combined_scores = scores + persistent_scores
                    
                    # Draw detections
                    result_frame = self.visualizer.draw_detections(
                        frame, combined_boxes, combined_scores, model_name
                    )
                    
                    # Add overlays
                    result_frame = self.visualizer.add_fps_overlay(result_frame, current_fps)
                    result_frame = self.visualizer.add_timestamp_overlay(result_frame, timestamp)
                    result_frame = self.visualizer.add_info_overlay(
                        result_frame, frame_count, len(combined_boxes), model_name
                    )
                    
                    # Save high-confidence detections
                    if self.saver:
                        self.saver.save_detection(
                            result_frame, boxes, scores, model_name, timestamp
                        )
                    
                    # Display
                    cv2.imshow('RPi5 Object Detection', result_frame)
                    
                    # Update FPS
                    fps_frame_count += 1
                    elapsed = time.time() - fps_start_time
                    if elapsed >= 1.0:
                        current_fps = fps_frame_count / elapsed
                        fps_frame_count = 0
                        fps_start_time = time.time()
                        
                        # Print stats every second
                        if self.saver:
                            print(f"FPS: {current_fps:.1f} | Frame: {frame_count} | "
                                  f"Detections: {len(combined_boxes)} | "
                                  f"Saved: {self.saver.get_save_count()}")
                        else:
                            print(f"FPS: {current_fps:.1f} | Frame: {frame_count} | "
                                  f"Detections: {len(combined_boxes)}")
                    
                    # Alternate models
                    if self.use_dual_models and config.USE_ALTERNATE_MODELS:
                        use_model_1 = not use_model_1
                
                if is_video and hasattr(self.camera, 'cap'):
                    video_fps = self.camera.cap.get(cv2.CAP_PROP_FPS)
                    if video_fps > 0:
                        frame_delay = int(1000 / video_fps)  # milliseconds per frame
                    else:
                        frame_delay = 33
                    key = cv2.waitKey(frame_delay) & 0xFF
                else:
                    key = cv2.waitKey(1) & 0xFF  


                if key == ord('q'):
                    print("\n⏹️  Stopped by user")
                    break
                elif key == ord('p'):
                    paused = not paused
                    print(f"{'⏸️  Paused' if paused else '▶️  Resumed'}")
                elif key == ord('s'):
                    # Manual save
                    if not paused and 'result_frame' in locals():
                        save_path = f"{config.OUTPUT_DIR}/manual_{frame_count}.jpg"
                        cv2.imwrite(save_path, result_frame)
                        print(f"💾 Manually saved: {save_path}")
                
        except KeyboardInterrupt:
            print("\n⏹️  Interrupted by user")
        
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Clean up resources"""
        print("\n" + "=" * 60)
        print("Cleaning up...")
        
        if self.camera:
            self.camera.release()
        
        if self.saver:
            self.saver.stop()
        
        cv2.destroyAllWindows()
        
        print("✓ Cleanup complete")
        print("=" * 60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='RPi5 Object Detection System - Optimized for high performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with camera (dual models)
  python main.py --source camera
  
  # Run with camera (single model)
  python main.py --source camera --single-model
  
  # Run with video file
  python main.py --source video.mp4
  
  # Custom camera with no saving
  python main.py --source camera --camera-id 1 --no-save
        """
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default='camera',
        help='Input source: "camera" or path to video file'
    )
    
    parser.add_argument(
        '--camera-id',
        type=int,
        default=0,
        help='Camera ID (default: 0)'
    )
    
    parser.add_argument(
        '--single-model',
        action='store_true',
        help='Use only anomaly model (no alternating)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Disable automatic saving of high-confidence detections'
    )
    
    parser.add_argument(
        '--conf-threshold',
        type=float,
        help='Override confidence threshold'
    )
    
    parser.add_argument(
        '--save-threshold',
        type=float,
        help='Override save threshold for high-confidence detections'
    )
    
    args = parser.parse_args()
    
    # Override config if specified
    if args.conf_threshold:
        config.ANOMALY_CONF_THRESHOLD = args.conf_threshold
        config.POTHOLE_CONF_THRESHOLD = args.conf_threshold
    
    if args.save_threshold:
        config.HIGH_CONF_SAVE_THRESHOLD = args.save_threshold
    
    # Initialize system
    try:
        system = RPi5DetectionSystem(
            use_dual_models=not args.single_model,
            save_detections=not args.no_save
        )
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        sys.exit(1)
    
    # Run detection
    try:
        if args.source.lower() == 'camera':
            system.run_camera(camera_id=args.camera_id)
        else:
            system.run_video(video_path=args.source)
    except Exception as e:
        print(f"❌ Error during detection: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
