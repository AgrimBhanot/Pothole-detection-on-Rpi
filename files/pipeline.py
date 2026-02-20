"""
Multiprocessing pipeline for async detection
Separates capture, inference, and display into parallel processes
"""

import multiprocessing as mp
import time
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class DetectionResult:
    """Container for detection results"""
    frame: np.ndarray
    boxes: List[Tuple[int, int, int, int]]
    scores: List[float]
    model_name: str
    timestamp: float
    frame_id: int
    
@dataclass
class FrameData:
    """Container for frame data"""
    frame: np.ndarray
    frame_id: int
    timestamp: float


class DetectionPipeline:
    """
    Multiprocessing pipeline for object detection
    
    Architecture:
    - Process A: Capture frames (camera/video)
    - Process B: Run inference (alternating models)
    - Process C: Display and save results
    
    This eliminates serial bottlenecks and maximizes throughput
    """
    
    def __init__(self, max_queue_size: int = 2, max_latency_ms: int = 200):
        """
        Initialize the pipeline
        
        Args:
            max_queue_size: Maximum frames in queue (small = low latency)
            max_latency_ms: Drop frames if latency exceeds this
        """
        # Queues for inter-process communication
        self.capture_queue = mp.Queue(maxsize=max_queue_size)
        self.detection_queue = mp.Queue(maxsize=max_queue_size)
        
        # Control flags
        self.running = mp.Value('i', 1)  # Shared integer (1=running, 0=stop)
        self.frame_counter = mp.Value('i', 0)  # Shared frame counter
        
        # Performance tracking
        self.max_latency_ms = max_latency_ms
        
        print("✓ Detection pipeline initialized")
    
    def start_capture_process(self, camera, target_fps: Optional[int] = None):
        """
        Start the capture process
        
        Args:
            camera: ThreadedCamera or VideoCapture instance
            target_fps: Target frame rate (None = unlimited)
        """
        def capture_worker():
            """Worker that captures frames and puts them in queue"""
            frame_delay = (1.0 / target_fps) if target_fps else 0.0
            last_time = time.time()
            
            while self.running.value:
                current_time = time.time()
                
                # Rate limiting
                if target_fps and (current_time - last_time) < frame_delay:
                    time.sleep(0.001)
                    continue
                
                ret, frame = camera.read()
                if not ret:
                    break
                
                # Get frame ID
                with self.frame_counter.get_lock():
                    frame_id = self.frame_counter.value
                    self.frame_counter.value += 1
                
                # Create frame data
                frame_data = FrameData(
                    frame=frame,
                    frame_id=frame_id,
                    timestamp=current_time
                )
                
                # Put in queue (non-blocking to check latency)
                try:
                    self.capture_queue.put(frame_data, block=False)
                    last_time = current_time
                except:
                    # Queue full - this means we're processing too slowly
                    # Drop this frame and continue
                    pass
        
        process = mp.Process(target=capture_worker, daemon=True)
        process.start()
        return process
    
    def start_inference_process(self, detector1, detector2, use_alternate: bool = True):
        """
        Start the inference process
        
        Args:
            detector1: First detector (e.g., Anomaly)
            detector2: Second detector (e.g., Pothole)
            use_alternate: Whether to alternate between models
        """
        def inference_worker():
            """Worker that runs detection and puts results in queue"""
            # Track which model to use
            use_model_1 = True
            
            # Persistence for boxes (to prevent flickering)
            last_boxes_1 = []
            last_scores_1 = []
            last_boxes_2 = []
            last_scores_2 = []
            
            while self.running.value:
                try:
                    # Get frame from queue (with timeout)
                    frame_data = self.capture_queue.get(timeout=0.1)
                except:
                    continue
                
                # Check latency
                latency_ms = (time.time() - frame_data.timestamp) * 1000
                if latency_ms > self.max_latency_ms:
                    # Frame is too old, drop it
                    continue
                
                # Select detector
                if use_alternate:
                    detector = detector1 if use_model_1 else detector2
                    model_name = detector.get_name()
                else:
                    detector = detector1
                    model_name = detector.get_name()
                
                # Run detection
                boxes, scores = detector.detect(frame_data.frame)
                
                # Store results for persistence
                if use_model_1:
                    last_boxes_1 = boxes
                    last_scores_1 = scores
                    # Use previous model 2 detections for continuity
                    persistent_boxes = last_boxes_2
                    persistent_scores = last_scores_2
                else:
                    last_boxes_2 = boxes
                    last_scores_2 = scores
                    # Use previous model 1 detections for continuity
                    persistent_boxes = last_boxes_1
                    persistent_scores = last_scores_1
                
                # Combine current and persistent detections
                combined_boxes = boxes + persistent_boxes
                combined_scores = scores + persistent_scores
                
                # Create result
                result = DetectionResult(
                    frame=frame_data.frame,
                    boxes=combined_boxes,
                    scores=combined_scores,
                    model_name=model_name,
                    timestamp=frame_data.timestamp,
                    frame_id=frame_data.frame_id
                )
                
                # Put in queue
                try:
                    self.detection_queue.put(result, block=False)
                except:
                    # Queue full, drop oldest result
                    try:
                        self.detection_queue.get_nowait()
                        self.detection_queue.put(result, block=False)
                    except:
                        pass
                
                # Alternate models
                if use_alternate:
                    use_model_1 = not use_model_1
        
        process = mp.Process(target=inference_worker, daemon=True)
        process.start()
        return process
    
    def get_result(self, timeout: float = 0.1) -> Optional[DetectionResult]:
        """
        Get detection result from queue
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            DetectionResult or None
        """
        try:
            return self.detection_queue.get(timeout=timeout)
        except:
            return None
    
    def stop(self):
        """Stop all processes"""
        self.running.value = 0
        
        # Clear queues
        while not self.capture_queue.empty():
            try:
                self.capture_queue.get_nowait()
            except:
                break
        
        while not self.detection_queue.empty():
            try:
                self.detection_queue.get_nowait()
            except:
                break
        
        print("✓ Pipeline stopped")
    
    def get_queue_sizes(self) -> Tuple[int, int]:
        """
        Get current queue sizes
        
        Returns:
            Tuple of (capture_queue_size, detection_queue_size)
        """
        return self.capture_queue.qsize(), self.detection_queue.qsize()
