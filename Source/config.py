"""
Configuration file for RPi5 Object Detection System
Centralised settings for models, performance tuning, detection parameters,
adaptive model-pair switching, and ByteTrack object tracking.
"""
import os
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ModelConfig:
    """Configuration for a single YOLO model."""
    path: str
    conf_threshold: float
    nms_threshold: float
    input_size: Tuple[int, int] = (416, 416)
    name: str = "Model"


@dataclass
class SystemConfig:
    """System-wide configuration."""

    # Model paths — Performance pair (FP32 / unquantized) 
    ANOMALY_MODEL_PATH: str = "new_model/General_FP32.onnx"
    POTHOLE_MODEL_PATH: str = "new_model/Pothole_FP32.onnx"

    # ~~ Model paths — Efficiency pair (INT8 quantized) ~~~~~~~~~~~~~~~~~~~
    # If these files are absent, Efficiency mode will reuse the FP32 pair.
    ANOMALY_MODEL_QUANT_PATH: str = "new_model/Model2_General.onnx"
    POTHOLE_MODEL_QUANT_PATH: str = "new_model/Model1_Pothole.onnx"

    # ~~ Detection thresholds ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ANOMALY_CONF_THRESHOLD: float  = 0.7
    POTHOLE_CONF_THRESHOLD: float  = 0.7
    HIGH_CONF_SAVE_THRESHOLD: float = 0.75
    NMS_THRESHOLD: float            = 0.3

    # ~~ ONNX Runtime thread settings for RPi5 (4-core) ~~~~~~~~~~~~~~~~~~~
    INTRA_OP_NUM_THREADS_EFF: int = 2
    INTER_OP_NUM_THREADS_EFF: int = 1
    INTRA_OP_NUM_THREADS_PERF: int = 4
    INTER_OP_NUM_THREADS_PERF: int = 1
    NUM_WARMUP_RUNS: int      = 10

    # ~~ Pipeline / queue settings ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Queue size of 1 → always drop stale frames, keep only the latest
    FRAME_QUEUE_SIZE: int  = 1
    MAX_LATENCY_MS: int    = 200   # Drop frames older than this
    USE_ALTERNATE_MODELS: bool = True

    # ~~ Adaptive model-pair switching — hysteresis thresholds ~~~~~~~~~~~~~
    #
    #   DOWNGRADE  PERFORMANCE → EFFICIENCY  when:
    #       EMA FPS  < FPS_DOWNGRADE_THRESHOLD   (8 fps)
    #       OR CPU temp > TEMP_DOWNGRADE_THRESHOLD (75 °C)
    #
    #   UPGRADE    EFFICIENCY  → PERFORMANCE  when:
    #       EMA FPS  > FPS_UPGRADE_THRESHOLD     (12 fps)
    #       AND CPU temp < TEMP_UPGRADE_THRESHOLD (65 °C)
    #
    #   The deliberate gap between down (8/75) and up (12/65) is the
    #   hysteresis dead-band that prevents mode oscillation.
    #
    FPS_DOWNGRADE_THRESHOLD:  float = 4.0 #8.0
    FPS_UPGRADE_THRESHOLD:    float = 7.0 #12.0
    TEMP_DOWNGRADE_THRESHOLD: float = 75.0   # °C
    TEMP_UPGRADE_THRESHOLD:   float = 65.0   # °C

    # Cooldown between successive switches (seconds)
    SWITCH_COOLDOWN_SECONDS: float = 10.0 #2.0

    # EMA smoothing factor for FPS (lower α → slower reaction, less noise)
    FPS_EMA_ALPHA: float = 0.1 #0.2

    # ~~ ByteTrack tracker settings ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    TRACKER_MAX_LOST_FRAMES:     int   = 15    # Frames before a lost track is removed
    TRACKER_IOU_THRESHOLD:       float = 0.35  # Min IoU for a valid match
    TRACKER_HIGH_CONF_THRESHOLD: float = 0.60  # High-confidence tier boundary
    TRACKER_LOW_CONF_THRESHOLD:  float = 0.30  # Low-confidence tier boundary

    # ~~ Camera settings ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    CAMERA_WIDTH:  int = 640
    CAMERA_HEIGHT: int = 480
    CAMERA_FPS:    int = 30

    # ~~ Output / display settings ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    SAVE_DETECTIONS: bool = True
    OUTPUT_DIR: str       = "detections"
    DISPLAY_FPS: bool     = True
    SHOW_TIMESTAMPS: bool = True

    # ~~ Video playback ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    VIDEO_PROCESS_ALL_FRAMES: bool = True
    VIDEO_DISPLAY_SPEED: str       = "original"

    # ~~ Colour schemes (BGR for OpenCV) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    COLOR_ANOMALY:  Tuple[int, int, int] = (0, 255, 0)    # Green
    COLOR_POTHOLE:  Tuple[int, int, int] = (0, 0, 255)    # Red
    COLOR_TEXT_BG:  Tuple[int, int, int] = (0, 0, 0)      # Black
    COLOR_TEXT:     Tuple[int, int, int] = (255, 255, 255) # White

    def __post_init__(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

    # ~~ ModelConfig factories ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_anomaly_config(self) -> ModelConfig:
        """FP32 general obstacle detector config."""
        return ModelConfig(
            path=self.ANOMALY_MODEL_PATH,
            conf_threshold=self.ANOMALY_CONF_THRESHOLD,
            nms_threshold=self.NMS_THRESHOLD,
            name="Model_General",
        )

    def get_pothole_config(self) -> ModelConfig:
        """FP32 pothole detector config."""
        return ModelConfig(
            path=self.POTHOLE_MODEL_PATH,
            conf_threshold=self.POTHOLE_CONF_THRESHOLD,
            nms_threshold=self.NMS_THRESHOLD,
            name="Model_Pothole",
        )

    def get_anomaly_quant_config(self) -> ModelConfig:
        """INT8 quantized general obstacle detector config."""
        return ModelConfig(
            path=self.ANOMALY_MODEL_QUANT_PATH,
            conf_threshold=self.ANOMALY_CONF_THRESHOLD,
            nms_threshold=self.NMS_THRESHOLD,
            name="Model_General_INT8",
        )

    def get_pothole_quant_config(self) -> ModelConfig:
        """INT8 quantized pothole detector config."""
        return ModelConfig(
            path=self.POTHOLE_MODEL_QUANT_PATH,
            conf_threshold=self.POTHOLE_CONF_THRESHOLD,
            nms_threshold=self.NMS_THRESHOLD,
            name="Model_Pothole_INT8",
        )


# Module-level singleton — imported everywhere
config = SystemConfig()
