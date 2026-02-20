"""
Runs camera capture on a background thread so the main loop always gets the freshest frame without waiting.
"""

# import cv2
# import threading
# import time
# from typing import Optional, Tuple
# import numpy as np

# class ThreadedCamera:
#     """
#     Camera capture with background thread
#     Ensures the camera buffer is always fresh and prevents lag
#     """
    
#     def __init__(self, src: int = 0, width: int = 640, height: int = 480, fps: int = 30):
#         """
#         Initialize threaded camera
        
#         Args:
#             src: Camera index or video path
#             width: Frame width
#             height: Frame height
#             fps: Target FPS
#         """
#         self.src = src
                
#                 # Check if src is a GStreamer pipeline string
#         if isinstance(src, str) and "!" in src:
#             self.cap = cv2.VideoCapture(src, cv2.CAP_GSTREAMER)
#         else:
#                     # Fallback for USB cameras (integers) or video files
#             self.cap = cv2.VideoCapture(src)
                
#         if not self.cap.isOpened():
#             raise RuntimeError(f"Failed to open camera/video: {src}")

#                 # Note: If using GStreamer, properties like width/height 
#                 # are usually set IN the pipeline string, but we can try setting them here too
#         if isinstance(src, int):
#             self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#             self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
#             self.cap.set(cv2.CAP_PROP_FPS, fps)
        
#         if not self.cap.isOpened():
#             raise RuntimeError(f"Failed to open camera/video: {src}")
        
#         # Set camera properties
#         self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#         self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
#         self.cap.set(cv2.CAP_PROP_FPS, fps)
        
#         # Enable hardware acceleration if available
#         self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
#         # Read first frame
#         ret, self.frame = self.cap.read()
#         if not ret:
#             raise RuntimeError("Failed to read first frame")
        
#         # Thread control
#         self.lock = threading.Lock()
#         self.running = True
#         self.thread = threading.Thread(target=self._update, daemon=True)
#         self.thread.start()
        
#         # Statistics
#         self.frame_count = 0
#         self.start_time = time.time()
        
#         print(f"✓ Camera initialized: {width}x{height} @ {fps}fps")
    
#     def _update(self):
#         """
#         Background thread that continuously reads frames
#         This prevents camera buffer from filling up with old frames
#         """
#         while self.running:
#             ret, frame = self.cap.read()
#             if ret:
#                 with self.lock:
#                     self.frame = frame
#                     self.frame_count += 1
#             else:
#                 # If reading fails, try to reconnect
#                 self.running = False
#                 print("Frame read failed, attempting to reconnect...")
#                 break
#             time.sleep(0.01)  # Small sleep to prevent CPU hogging    
#     def read(self) -> Tuple[bool, Optional[np.ndarray]]:
#         """
#         Get the latest frame
        
#         Returns:
#             Tuple of (success, frame)
#         """
#         with self.lock:
#             return True, self.frame.copy()
    
#     def get_fps(self) -> float:
#         """
#         Calculate actual capture FPS
        
#         Returns:
#             Frames per second
#         """
#         elapsed = time.time() - self.start_time
#         if elapsed > 0:
#             return self.frame_count / elapsed
#         return 0.0
    
#     def release(self):
#         """Stop the thread and release camera"""
#         self.running = False
#         self.thread.join(timeout=2.0)
#         self.cap.release()
#         print("✓ Camera released")
import cv2
import threading
import time
from typing import Optional, Tuple
import numpy as np
from picamera2 import Picamera2 # Ensure this is installed

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
        self.frame = None
        
        # Determine if we use Picamera2 (Pi 5 CSI) or OpenCV (USB/Files)
        if src == "pi_camera" or src == 0:
            print("🚀 Initializing RPi5 CSI Camera via Picamera2...")
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"size": (width, height), "format": "BGR888"}
            )
            self.picam2.configure(config)
            self.picam2.start()
            self.is_pi_cam = True
        else:
            print(f"🔌 Initializing standard source: {src}")
            self.cap = cv2.VideoCapture(src)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            self.cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))##########################
            self.cap.set(cv2.CAP_PROP_AUTO_WB,1)
            self.is_pi_cam = False

        # Read first frame to verify and initialize self.frame
        if self.is_pi_cam:
            self.frame = self.picam2.capture_array()
        else:
            ret, self.frame = self.cap.read()
            if not ret:
                raise RuntimeError("Failed to read first frame from source")

        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        self.frame_count = 0
        self.start_time = time.time()
        
        print(f" Camera initialized: {width}x{height} @ {fps}fps")
    
    def _update(self):
        while self.running:
            try:
                if self.is_pi_cam:
                    # Capture from Pi Camera
                    raw_frame = self.picam2.capture_array()
                    # RPi5 often sends RGB even if BGR is requested. 
                    # Swap them here to fix the bluish tint.
                    frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
                else:
                    ret, frame = self.cap.read()
                    if not ret: 
                        break
                
                with self.lock:
                    self.frame = frame
                    self.frame_count += 1
            except Exception as e:
                print(f"Capture error: {e}")
                break
            time.sleep(0.01)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def is_opened(self) -> bool:
        return self.running

    def release(self):
        self.running = False
        self.thread.join(timeout=2.0)
        if self.is_pi_cam:
            self.picam2.stop()
        else:
            self.cap.release()
    
    def is_opened(self) -> bool:
        """Check if camera is still open"""
        if self.is_pi_cam:
            return self.running
        else:
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
        # Video files use self.cap, never picamera2
        if hasattr(self, 'cap') and self.total_frames > 0:
            current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            return (current_frame / self.total_frames) * 100
        return 0.0

    def is_finished(self) -> bool:
        if hasattr(self, 'cap'):
            current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            return current_frame >= self.total_frames - 1
        return True
