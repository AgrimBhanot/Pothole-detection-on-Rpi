"""
Visualization and image saving utilities
Handles drawing boxes, FPS display, Info overlays,  Timestamps and high-confidence image saving
"""
import cv2
import numpy as np
import os
from datetime import datetime
from typing import List, Tuple, Optional
from threading import Thread, Lock
from queue import Queue

class Visualizer:
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
        
        self.fps_buffer = []
        self.fps_buffer_size = 30
        
    def draw_detections(self, frame: np.ndarray, boxes: List[Tuple[int, int, int, int]],
                       scores: List[float], model_name: str) -> np.ndarray:
      
        
        color = self.color_pothole if "Pothole" in model_name else self.color_anomaly
        
        for (x1, y1, x2, y2), score in zip(boxes, scores):
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{model_name}: {score:.2f}"
            
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(
                frame,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1
            )
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
       
        if not self.show_fps:
            return frame
        
        self.fps_buffer.append(fps)
        if len(self.fps_buffer) > self.fps_buffer_size:
            self.fps_buffer.pop(0)
        
        avg_fps = sum(self.fps_buffer) / len(self.fps_buffer)
        
        fps_text = f"FPS: {avg_fps:.1f}"
        
        (text_width, text_height), baseline = cv2.getTextSize(
            fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )
        cv2.rectangle(
            frame,
            (10, 10),
            (20 + text_width, 20 + text_height),
            (0, 0, 0),
            -1
        )
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
        
        info_text = f"Frame: {frame_id} | Detections: {num_detections} | Model: {model_name}"
        
        (text_width, text_height), baseline = cv2.getTextSize(
            info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        cv2.rectangle(
            frame,
            (0, h - text_height - 20),
            (text_width + 20, h),
            (0, 0, 0),
            -1
        )
        
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

        if not self.show_timestamp:
            return frame
        
        dt = datetime.fromtimestamp(timestamp)
        time_text = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        (text_width, text_height), baseline = cv2.getTextSize(
            time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        
        h, w = frame.shape[:2]
        
        cv2.rectangle(
            frame,
            (w - text_width - 20, 10),
            (w - 10, 20 + text_height),
            (0, 0, 0),
            -1
        )
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
        
        self.output_dir = output_dir
        self.high_conf_threshold = high_conf_threshold
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.save_queue = Queue()
        self.lock = Lock()
        self.running = True
        self.save_count = 0
        
        self.thread = Thread(target=self._save_worker, daemon=True)
        self.thread.start()
        
        print(f" Detection saver initialized (threshold: {high_conf_threshold})")
    
    def _save_worker(self):
        """Background worker that saves images"""
        while self.running:
            try:
                task = self.save_queue.get(timeout=0.5)
                if task is None:
                    break
                
                frame, timestamp, model_name, max_conf = task
                
                dt = datetime.fromtimestamp(timestamp)
                filename = f"{model_name}_{dt.strftime('%Y%m%d_%H%M%S_%f')}_{max_conf:.2f}.jpg"
                filepath = os.path.join(self.output_dir, filename)
                
                cv2.imwrite(filepath, frame)
                
                with self.lock:
                    self.save_count += 1
                
            except:
                continue
    
    def save_detection(self, frame: np.ndarray, boxes: List[Tuple[int, int, int, int]],
                      scores: List[float], model_name: str, timestamp: float):
        
        if not scores:
            return
        
        max_conf = max(scores)
        
        if max_conf >= self.high_conf_threshold:
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
        print(f" Detection saver stopped ({self.save_count} images saved)")
