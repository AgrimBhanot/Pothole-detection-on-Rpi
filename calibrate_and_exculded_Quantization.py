import onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
import numpy as np
import cv2
import os

# 1. D
class YOLODataCalibrationReader(CalibrationDataReader):
    def __init__(self, image_folder, input_name):
        import os, glob
        self.input_name = input_name
        self.image_paths = sorted(glob.glob(os.path.join(image_folder, "*")))
        self.count = 0

    def _preprocess(self, path):
        import cv2, numpy as np
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        img = cv2.resize(img, (416, 416))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return {self.input_name: img}

    def get_next(self):
        while self.count < len(self.image_paths):
            path = self.image_paths[self.count]
            self.count += 1
            try:
                img = self._preprocess(path)
                print("Calibration step:", self.count)
                return img
            except ValueError:
                print(f"Skipping invalid image: {path}")
        return None

nodes_to_skip = [
    # '/model.22/cv2.0/cv2.0.2/Conv', 
    '/model.22/dfl/conv/Conv',#
    '/model.22/cv3.1/cv3.1.2/Conv',
    '/model.22/cv2.1/cv2.1.2/Conv',
    '/model.22/cv2.2/cv2.2.2/Conv',
    '/model.22/cv3.2/cv3.2.2/Conv',
    '/model.22/Concat_4',#

    '/model.22/Concat_5', # Often the final assembly point
]

quantize_static(
    model_input="new_model/best_preprocessed.onnx",
    model_output="new_model/new_preprocessed_excluded.onnx",
    calibration_data_reader=YOLODataCalibrationReader(
        "calibrate",
        "images"
    ),
    quant_format=QuantType.QInt8,
    per_channel=True,
    weight_type=QuantType.QInt8,
    nodes_to_exclude=nodes_to_skip
)
