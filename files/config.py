"""
Configuration file for RPi5 Object Detection System
Centralized settings for models, performance tuning, and detection parameters
"""

import os
from dataclasses import dataclass
from typing import Tuple

@dataclass
class ModelConfig:
    """Configuration for individual YOLO models"""
    path: str
    conf_threshold: float
    nms_threshold: float
    input_size: Tuple[int, int] = (416, 416)
    name: str = "Model"
    
@dataclass
class SystemConfig:
    """System-wide configuration"""
    # Model paths
    ANOMALY_MODEL_PATH: str = "files/new_preprocessed_excluded.onnx"
    POTHOLE_MODEL_PATH: str = "files/best_preprocessed_excluded.onnx"  # Your second model
    
    # Detection thresholds
    ANOMALY_CONF_THRESHOLD: float = 0.5
    POTHOLE_CONF_THRESHOLD: float = 0.5
    HIGH_CONF_SAVE_THRESHOLD: float = 0.75  # Save images above this confidence
    NMS_THRESHOLD: float = 0.45
    
    # Performance settings for RPi5
    INTRA_OP_NUM_THREADS: int = 4  # Pi5 has 4 cores
    INTER_OP_NUM_THREADS: int = 2
    NUM_WARMUP_RUNS: int = 10
    VIDEO_PROCESS_ALL_FRAMES: bool = True  # Process every frame in video
    VIDEO_DISPLAY_SPEED: str = "original"
    # Pipeline settings
    FRAME_QUEUE_SIZE: int = 10  # Small queue to minimize latency
    MAX_LATENCY_MS: int = 1000  # Drop frames if latency exceeds this
    USE_ALTERNATE_MODELS: bool = True  # Alternate between models each frame
    
    # Camera settings
    CAMERA_WIDTH: int = 640
    CAMERA_HEIGHT: int = 480
    CAMERA_FPS: int = 30
    
    # Output settings
    SAVE_DETECTIONS: bool = True
    OUTPUT_DIR: str = "detections"
    DISPLAY_FPS: bool = True
    SHOW_TIMESTAMPS: bool = True
    
    # Color schemes (BGR format for OpenCV)
    COLOR_ANOMALY: Tuple[int, int, int] = (0, 255, 0)  # Green
    COLOR_POTHOLE: Tuple[int, int, int] = (0, 0, 255)  # Red
    COLOR_TEXT_BG: Tuple[int, int, int] = (0, 0, 0)  # Black
    COLOR_TEXT: Tuple[int, int, int] = (255, 255, 255)  # White
    
    def __post_init__(self):
        """Create output directory if it doesn't exist"""
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        
    def get_anomaly_config(self) -> ModelConfig:
        """Get configuration for anomaly detection model"""
        return ModelConfig(
            path=self.ANOMALY_MODEL_PATH,
            conf_threshold=self.ANOMALY_CONF_THRESHOLD,
            nms_threshold=self.NMS_THRESHOLD,
            name="Anomaly"
        )
    
    def get_pothole_config(self) -> ModelConfig:
        """Get configuration for pothole detection model"""
        return ModelConfig(
            path=self.POTHOLE_MODEL_PATH,
            conf_threshold=self.POTHOLE_CONF_THRESHOLD,
            nms_threshold=self.NMS_THRESHOLD,
            name="Pothole"
        )

# Global configuration instance
config = SystemConfig()
