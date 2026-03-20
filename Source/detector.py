"""
Optimised YOLO Detector for Raspberry Pi 5
===========================================
Key features:
  • Vectorised pre/postprocessing (NumPy)
  • OpenCV NMS  — cv2.dnn.NMSBoxes() (Task 4 ✓ already implemented)
  • Pre-allocated input buffer (zero-copy preprocessing)
  • ONNX Runtime with all graph optimisations enabled

NOTE (Task 4): Python NMS has NOT been used here — cv2.dnn.NMSBoxes()
was already the implementation in the original codebase.  No change needed.
"""

import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple
from config import ModelConfig


class OptimizedYOLODetector:
    def __init__(
        self,
        model_config:  ModelConfig,
        intra_threads: int = 2, #4,
        inter_threads: int = 1, #2,
        warmup_runs:   int = 10,
    ) -> None:

        self.model_path    = model_config.path
        self.conf_threshold = model_config.conf_threshold
        self.nms_threshold  = model_config.nms_threshold
        self.input_width, self.input_height = model_config.input_size
        self.name = model_config.name

        # Configure ONNX Runtime session
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads  = intra_threads
        opts.inter_op_num_threads  = inter_threads
        opts.execution_mode        = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.enable_profiling      = False

        # self.session    = ort.InferenceSession(
        #     self.model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        # )

        providers = []
        available = ort.get_available_providers()
        if "XNNPACKExecutionProvider" in available:
            providers.append(("XNNPACKExecutionProvider", {"intra_op_num_threads": str(intra_threads)}))
            print(f"[{self.name}] Using XNNPACK provider")
        providers.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(self.model_path, sess_options=opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        # Pre-allocated inference buffer — avoids per-frame allocation
        self.input_buffer = np.zeros(
            (1, 3, self.input_height, self.input_width), dtype=np.float32
        )

        print(f"  ✓ {self.name} loaded: {self.model_path}")
        self._warmup(warmup_runs)

    # ── Warmup ────────────────────────────────────────────────────────────

    def _warmup(self, num_runs: int) -> None:
        print(f"   Warming up {self.name} ({num_runs} runs)…", end=" ", flush=True)
        dummy = np.random.rand(
            1, 3, self.input_height, self.input_width
        ).astype(np.float32)
        for _ in range(num_runs):
            self.session.run(None, {self.input_name: dummy})
        print("done.")

    # ── Preprocessing ─────────────────────────────────────────────────────

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize, normalise, and transpose to NCHW in a pre-allocated buffer.
        Avoids per-frame memory allocation (zero-copy optimisation).
        """
        img = cv2.resize(frame, (self.input_width, self.input_height))
        img_f = img.astype(np.float32) * (1.0 / 255.0)

        # Transpose HWC → CHW directly into buffer
        self.input_buffer[0, 0] = img_f[:, :, 0]   # B channel
        self.input_buffer[0, 1] = img_f[:, :, 1]   # G channel
        self.input_buffer[0, 2] = img_f[:, :, 2]   # R channel
        return self.input_buffer

    # ── Postprocessing ────────────────────────────────────────────────────

    def postprocess(
        self,
        outputs:     List[np.ndarray],
        frame_shape: Tuple[int, int],
    ) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:

        h, w = frame_shape

        # YOLOv8 output layout: [1, 84, 8400] → squeeze → transpose → [8400, 84]
        predictions = np.squeeze(outputs[0]).T

        class_scores = predictions[:, 4:]
        scores       = np.max(class_scores, axis=1)
        mask         = scores > self.conf_threshold

        if not mask.any():
            return [], []

        filtered = predictions[mask]
        scores   = scores[mask]
        boxes    = filtered[:, :4]

        # Scale from model input space to original frame space
        sx = w / self.input_width
        sy = h / self.input_height

        # Centre-format → corner-format
        x1 = np.clip((boxes[:, 0] - boxes[:, 2] / 2) * sx, 0, w).astype(np.int32)
        y1 = np.clip((boxes[:, 1] - boxes[:, 3] / 2) * sy, 0, h).astype(np.int32)
        x2 = np.clip((boxes[:, 0] + boxes[:, 2] / 2) * sx, 0, w).astype(np.int32)
        y2 = np.clip((boxes[:, 1] + boxes[:, 3] / 2) * sy, 0, h).astype(np.int32)

        if len(x1) == 0:
            return [], []

        # ── Task 4: OpenCV NMS (C++ backend, no Python loops) ─────────────
        boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
        indices    = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(),
            scores.tolist(),
            score_threshold = self.conf_threshold,
            nms_threshold   = self.nms_threshold,
        )

        if len(indices) == 0:
            return [], []

        indices = indices.flatten()
        return (
            [(x1[i], y1[i], x2[i], y2[i]) for i in indices],
            [float(scores[i])              for i in indices],
        )

    # ── Full detect ───────────────────────────────────────────────────────

    def detect(
        self, frame: np.ndarray
    ) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
        """Run preprocess → inference → postprocess on a single frame."""
        img_data       = self.preprocess(frame)
        outputs        = self.session.run(None, {self.input_name: img_data})
        boxes, scores  = self.postprocess(outputs, frame.shape[:2])
        return boxes, scores

    def get_name(self) -> str:
        return self.name
