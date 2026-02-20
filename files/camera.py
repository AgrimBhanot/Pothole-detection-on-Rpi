"""
Threaded camera capture for continuous frame grabbing
Prevents camera buffer lag and ensures fresh frames
"""

import cv2
import threading
import time
from typing import Optional, Tuple
import numpy as np

class ThreadedCamera:
    """
    Camera capture with background thread
    Ensures the camera buffer is always fresh and prevents lag
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
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        # Enable hardware acceleration if available
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # Read first frame
        ret, self.frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read first frame")
        
        # Thread control
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        
        # Statistics
        self.frame_count = 0
        self.start_time = time.time()
        
        print(f"✓ Camera initialized: {width}x{height} @ {fps}fps")
    
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
                # If reading fails, try to reconnect
                print("⚠️  Frame read failed, attempting to reconnect...")
                time.sleep(0.1)
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Get the latest frame
        
        Returns:
            Tuple of (success, frame)
        """
        with self.lock:
            return True, self.frame.copy()
    
    def get_fps(self) -> float:
        """
        Calculate actual capture FPS
        
        Returns:
            Frames per second
        """
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            return self.frame_count / elapsed
        return 0.0
    
    def release(self):
        """Stop the thread and release camera"""
        self.running = False
        self.thread.join(timeout=2.0)
        self.cap.release()
        print("✓ Camera released")
    
    def is_opened(self) -> bool:
        """Check if camera is still open"""
        return self.cap.isOpened() and self.running
    
    def get_frame_size(self) -> Tuple[int, int]:
        """Get current frame dimensions (width, height)"""
        with self.lock:
            h, w = self.frame.shape[:2]
            return w, h


class VideoCapture(ThreadedCamera):
    """
    Video file capture with threading
    Extends ThreadedCamera for video files
    """
    
    def __init__(self, video_path: str):
        """
        Initialize video capture
        
        Args:
            video_path: Path to video file
        """
        self.video_path = video_path
        temp_cap = cv2.VideoCapture(video_path)
        
        if not temp_cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        # Get video properties
        width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(temp_cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        temp_cap.release()
        
        # Initialize parent class
        super().__init__(src=video_path, width=width, height=height, fps=fps)
        
        print(f"✓ Video loaded: {self.total_frames} frames @ {fps}fps")
    
    def get_progress(self) -> float:
        """
        Get video playback progress
        
        Returns:
            Progress as percentage (0-100)
        """
        current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        if self.total_frames > 0:
            return (current_frame / self.total_frames) * 100
        return 0.0
    
    def is_finished(self) -> bool:
        """Check if video has finished playing"""
        current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        return current_frame >= self.total_frames - 1
