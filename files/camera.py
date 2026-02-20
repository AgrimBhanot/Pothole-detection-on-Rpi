"""
Runs camera capture on a background thread so the main loop always gets the freshest frame without waiting.
"""

import cv2
import threading
import time
from typing import Optional, Tuple
import numpy as np

class ThreadedCamera:
    """
    Wraps OpenCV capture in a thread so stale frames never pile up in the buffer.
    """
    
    def __init__(self, src: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        """
        Initialize threaded camera
        Args:
            src: Camera index or video path
            width: Frame width
            height: Frame height
            fps: Target FPS
        """
        self.src = src
        self.cap = cv2.VideoCapture(src)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera/video: {src}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret, self.frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read first frame")
        
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        self.frame_count = 0
        self.start_time = time.time()
        
        print(f" Camera initialized: {width}x{height} @ {fps}fps")
    
    def _update(self):
        """
        Background thread that continuously reads frames
        This prevents camera buffer from filling up with old frames
        """
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
                    self.frame_count += 1
            else:
                print("  Frame read failed, attempting to reconnect...")
                time.sleep(0.1)
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        
        with self.lock:
            return True, self.frame.copy()
    
    def get_fps(self) -> float:
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            return self.frame_count / elapsed
        return 0.0
    
    def release(self):
        """Stop the thread and release camera"""
        self.running = False
        self.thread.join(timeout=2.0)
        self.cap.release()
        print(" Camera released")
    
    def is_opened(self) -> bool:
        
        return self.cap.isOpened() and self.running
    
    def get_frame_size(self) -> Tuple[int, int]:
        with self.lock:
            h, w = self.frame.shape[:2]
            return w, h


class VideoCapture(ThreadedCamera):
    """
    Video file capture with threading
    Extends ThreadedCamera for video files
    """
    
    def __init__(self, video_path: str):
        
        self.video_path = video_path
        temp_cap = cv2.VideoCapture(video_path)
        
        if not temp_cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(temp_cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        temp_cap.release()
        
        super().__init__(src=video_path, width=width, height=height, fps=fps)
        
        print(f" Video loaded: {self.total_frames} frames @ {fps}fps")
    
    def get_progress(self) -> float:
        
        current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        if self.total_frames > 0:
            return (current_frame / self.total_frames) * 100
        return 0.0
    
    def is_finished(self) -> bool:
        current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        return current_frame >= self.total_frames - 1
