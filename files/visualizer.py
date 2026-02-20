"""
Visualization and image saving utilities
Handles drawing boxes, FPS display, and high-confidence image saving
"""

import cv2
import numpy as np
import os
from datetime import datetime
from typing import List, Tuple, Optional
from threading import Thread, Lock
from queue import Queue

class Visualizer:
    """
    Handles all visualization tasks
    - Drawing bounding boxes
    - FPS display
    - Timestamps
    - Info overlays
    """
    
    def __init__(self, color_anomaly: Tuple[int, int, int] = (0, 255, 0),
                 color_pothole: Tuple[int, int, int] = (0, 0, 255),
                 show_fps: bool = True,
                 show_timestamp: bool = True):
        """
        Initialize visualizer
        
        Args:
            color_anomaly: BGR color for anomaly detections
            color_pothole: BGR color for pothole detections
            show_fps: Whether to display FPS
            show_timestamp: Whether to display timestamps
        """
        self.color_anomaly = color_anomaly
        self.color_pothole = color_pothole
        self.show_fps = show_fps
        self.show_timestamp = show_timestamp
        
        # FPS calculation
        self.fps_buffer = []
        self.fps_buffer_size = 30
        
    def draw_detections(self, frame: np.ndarray, boxes: List[Tuple[int, int, int, int]],
                       scores: List[float], model_name: str) -> np.ndarray:
        """
        Draw bounding boxes on frame
        
        Args:
            frame: Input frame
            boxes: List of boxes (x1, y1, x2, y2)
            scores: List of confidence scores
            model_name: Name of the model that made the detection
            
        Returns:
            Frame with drawn boxes
        """
        # Select color based on model
        color = self.color_pothole if "Pothole" in model_name else self.color_anomaly
        
        for (x1, y1, x2, y2), score in zip(boxes, scores):
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = f"{model_name}: {score:.2f}"
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # Draw background for text
            cv2.rectangle(
                frame,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2
            )
        
        return frame
    
    def add_fps_overlay(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """
        Add FPS overlay to frame
        
        Args:
            frame: Input frame
            fps: Current FPS
            
        Returns:
            Frame with FPS overlay
        """
        if not self.show_fps:
            return frame
        
        # Update FPS buffer for smoothing
        self.fps_buffer.append(fps)
        if len(self.fps_buffer) > self.fps_buffer_size:
            self.fps_buffer.pop(0)
        
        # Calculate smoothed FPS
        avg_fps = sum(self.fps_buffer) / len(self.fps_buffer)
        
        # Create FPS text
        fps_text = f"FPS: {avg_fps:.1f}"
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )
        
        # Draw background
        cv2.rectangle(
            frame,
            (10, 10),
            (20 + text_width, 20 + text_height),
            (0, 0, 0),
            -1
        )
        
        # Draw text
        cv2.putText(
            frame,
            fps_text,
            (15, 15 + text_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        
        return frame
    
    def add_info_overlay(self, frame: np.ndarray, frame_id: int, 
                        num_detections: int, model_name: str) -> np.ndarray:
        """
        Add info overlay (frame number, detections, model)
        
        Args:
            frame: Input frame
            frame_id: Frame number
            num_detections: Number of detections
            model_name: Active model name
            
        Returns:
            Frame with info overlay
        """
        h, w = frame.shape[:2]
        
        # Create info text
        info_text = f"Frame: {frame_id} | Detections: {num_detections} | Model: {model_name}"
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # Draw background (bottom of frame)
        cv2.rectangle(
            frame,
            (0, h - text_height - 20),
            (text_width + 20, h),
            (0, 0, 0),
            -1
        )
        
        # Draw text
        cv2.putText(
            frame,
            info_text,
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        return frame
    
    def add_timestamp_overlay(self, frame: np.ndarray, timestamp: float) -> np.ndarray:
        """
        Add timestamp overlay
        
        Args:
            frame: Input frame
            timestamp: Unix timestamp
            
        Returns:
            Frame with timestamp overlay
        """
        if not self.show_timestamp:
            return frame
        
        # Format timestamp
        dt = datetime.fromtimestamp(timestamp)
        time_text = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        
        h, w = frame.shape[:2]
        
        # Draw background (top right)
        cv2.rectangle(
            frame,
            (w - text_width - 20, 10),
            (w - 10, 20 + text_height),
            (0, 0, 0),
            -1
        )
        
        # Draw text
        cv2.putText(
            frame,
            time_text,
            (w - text_width - 15, 15 + text_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )
        
        return frame


class DetectionSaver:
    """
    Asynchronous image saver for high-confidence detections
    Uses background thread to prevent blocking
    """
    
    def __init__(self, output_dir: str, high_conf_threshold: float = 0.75):
        """
        Initialize saver
        
        Args:
            output_dir: Directory to save images
            high_conf_threshold: Minimum confidence to save image
        """
        self.output_dir = output_dir
        self.high_conf_threshold = high_conf_threshold
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Queue and thread for async saving
        self.save_queue = Queue()
        self.lock = Lock()
        self.running = True
        self.save_count = 0
        
        # Start background thread
        self.thread = Thread(target=self._save_worker, daemon=True)
        self.thread.start()
        
        print(f"✓ Detection saver initialized (threshold: {high_conf_threshold})")
    
    def _save_worker(self):
        """Background worker that saves images"""
        while self.running:
            try:
                # Get save task from queue
                task = self.save_queue.get(timeout=0.5)
                if task is None:
                    break
                
                frame, timestamp, model_name, max_conf = task
                
                # Create filename with timestamp
                dt = datetime.fromtimestamp(timestamp)
                filename = f"{model_name}_{dt.strftime('%Y%m%d_%H%M%S_%f')}_{max_conf:.2f}.jpg"
                filepath = os.path.join(self.output_dir, filename)
                
                # Save image
                cv2.imwrite(filepath, frame)
                
                with self.lock:
                    self.save_count += 1
                
            except:
                continue
    
    def save_detection(self, frame: np.ndarray, boxes: List[Tuple[int, int, int, int]],
                      scores: List[float], model_name: str, timestamp: float):
        """
        Save detection if confidence exceeds threshold
        
        Args:
            frame: Frame with drawn boxes
            boxes: Detection boxes
            scores: Confidence scores
            model_name: Model name
            timestamp: Timestamp
        """
        if not scores:
            return
        
        # Get maximum confidence
        max_conf = max(scores)
        
        # Only save if confidence exceeds threshold
        if max_conf >= self.high_conf_threshold:
            # Add to queue for async saving
            self.save_queue.put((frame.copy(), timestamp, model_name, max_conf))
    
    def get_save_count(self) -> int:
        """Get total number of saved images"""
        with self.lock:
            return self.save_count
    
    def stop(self):
        """Stop the saver thread"""
        self.running = False
        self.save_queue.put(None)
        self.thread.join(timeout=2.0)
        print(f"✓ Detection saver stopped ({self.save_count} images saved)")
