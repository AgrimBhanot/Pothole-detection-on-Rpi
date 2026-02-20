"""
Optimized YOLO Detector for Raspberry Pi 5
The key features include: vectorized operations, NMS, Pre-allocated buffers for zero-copy operations and ONNX Runtime optimizations
"""

import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple
from config import ModelConfig

class OptimizedYOLODetector:
    def __init__(self, model_config: ModelConfig, intra_threads: int = 4, 
                 inter_threads: int = 2, warmup_runs: int = 10):
        """
        Initializing the detector with optimized settings for Raspberry Pi 5
        Args:
            model_config: Model configuration object
            intra_threads: Number of threads for intra-op parallelism
            inter_threads: Number of threads for inter-op parallelism
            warmup_runs: Number of dummy runs for cache warming
        """
        self.model_path = model_config.path
        self.conf_threshold = model_config.conf_threshold
        self.nms_threshold = model_config.nms_threshold
        self.input_width, self.input_height = model_config.input_size
        self.name = model_config.name
        
        # Configuring ONNX Runtime for RPi5
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = intra_threads
        sess_options.inter_op_num_threads = inter_threads
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.enable_profiling = False
        
        # Creating the session
        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        
        # Pre-allocating the preprocessing buffer (zero-copy optimization)
        self.input_buffer = np.zeros(
            (1, 3, self.input_height, self.input_width), 
            dtype=np.float32
        )
        
        print(f" {self.name} Model loaded: {self.model_path}")
        
        # For Warming up the model
        self._warmup(warmup_runs)
    
    def _warmup(self, num_runs: int):
        """
        Warming up the model with dummy inferences
        This initializes CPU caches and ONNX Runtime for stable performance
        """
        print(f" Warming up {self.name} model ({num_runs} runs)...", end=" ", flush=True)
        dummy_input = np.random.rand(1, 3, self.input_height, self.input_width).astype(np.float32)
        for _ in range(num_runs):
            self.session.run(None, {self.input_name: dummy_input})
        print("Done!")
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame with zero-copy optimization
        Writes directly into pre-allocated buffer
        
        Args:
            frame: Input frame (HxWxC BGR)
            
        Returns:
            Preprocessed image tensor (1xCxHxW)
        """
        # Resize frame
        img = cv2.resize(frame, (self.input_width, self.input_height))
        
        # Normalize and transpose in one operation (faster)
        img_float = img.astype(np.float32) * (1.0 / 255.0)
        
        # Write directly to pre-allocated buffer (zero-copy)
        # Transpose: HWC -> CHW
        self.input_buffer[0, 0, :, :] = img_float[:, :, 0]  # B
        self.input_buffer[0, 1, :, :] = img_float[:, :, 1]  # G
        self.input_buffer[0, 2, :, :] = img_float[:, :, 2]  # R
        
        return self.input_buffer
    
    def postprocess(self, outputs: List[np.ndarray], frame_shape: Tuple[int, int]) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
        """
        Vectorized postprocessing with NMS
        Args:
            outputs: Model outputs from ONNX Runtime
            frame_shape: Original frame shape (height, width)
        Returns:
            Tuple of (boxes, scores) where boxes are (x1, y1, x2, y2)
        """
        h, w = frame_shape
        # YOLOv8 output: [1, 84, 8400] or [1, num_classes+4, 8400]
        predictions = np.squeeze(outputs[0]).T  # [8400, 84] or [8400, 5]
        # Vectorized confidence filtering
        class_scores = predictions[:, 4:]
        scores = np.max(class_scores, axis=1)
        mask = scores > self.conf_threshold
        
        if not mask.any():
            return [], []
        
        # Filter predictions
        filtered = predictions[mask]
        scores = scores[mask]
        
        # Extract box coordinates
        boxes = filtered[:, :4]
        
        # Vectorized scaling to original image size
        scale_x = w / self.input_width
        scale_y = h / self.input_height
        
        # Converting from center format to corner format and scale
        x1 = ((boxes[:, 0] - boxes[:, 2] / 2) * scale_x)
        y1 = ((boxes[:, 1] - boxes[:, 3] / 2) * scale_y)
        x2 = ((boxes[:, 0] + boxes[:, 2] / 2) * scale_x)
        y2 = ((boxes[:, 1] + boxes[:, 3] / 2) * scale_y)
        
        # Clamping to image boundaries
        x1 = np.clip(x1, 0, w).astype(np.int32)
        y1 = np.clip(y1, 0, h).astype(np.int32)
        x2 = np.clip(x2, 0, w).astype(np.int32)
        y2 = np.clip(y2, 0, h).astype(np.int32)
        
        if len(x1) == 0:
            return [], []
        
        # Applying NMS to remove duplicate detections
        # Converting to xywh format for cv2.dnn.NMSBoxes
        boxes_nms = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
        
        indices = cv2.dnn.NMSBoxes(
            boxes_nms.tolist(),
            scores.tolist(),
            score_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold
        )
        
        if len(indices) == 0:
            return [], []
        
        # Flattening indices 
        indices = indices.flatten()
        
        # Building final results
        final_boxes = [(x1[i], y1[i], x2[i], y2[i]) for i in indices]
        final_scores = [scores[i] for i in indices]
        return final_boxes, final_scores
    
    def detect(self, frame: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
        """
        For running a full detection pipeline on a frame
        Args:
            frame: Input frame (HxWxC BGR)
        Returns:
            Tuple of (boxes, scores)
        """
        # Preprocessing
        img_data = self.preprocess(frame)
        
        # Running inference
        outputs = self.session.run(None, {self.input_name: img_data})
        
        # Postprocessing
        boxes, scores = self.postprocess(outputs, frame.shape[:2])
        
        return boxes, scores
    def get_name(self) -> str:
        """Get model name"""
        return self.name
