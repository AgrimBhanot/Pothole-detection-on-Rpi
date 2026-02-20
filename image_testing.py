import cv2
import numpy as np
import onnxruntime as ort
import os

# 1. Setup Session
# On x64 laptops, we don't need the specific ARM threading tweaks
model_path = "new_model/new_preprocessed_excluded.onnx"
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# 2. Path Setup
image_folder = "test_images"  # Folder where you downloaded images
output_folder = "results"
os.makedirs(output_folder, exist_ok=True)

# 3. Processing Loop
for img_name in os.listdir(image_folder):
    img_path = os.path.join(image_folder, img_name)
    orig_frame = cv2.imread(img_path)
    if orig_frame is None: continue
    
    # Pre-process (must match your calibration logic)
    h, w = orig_frame.shape[:2]
    img = cv2.resize(orig_frame, (416, 416))
    img_data = img.astype(np.float32) / 255.0
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0)
    
    # Run Inference
    outputs = session.run(None, {input_name: img_data})
    
    # YOLOv8 Output is typically [1, 84, 8400]
    # 84 = 4 box coords + 80 class scores (or however many classes you have)
    predictions = np.squeeze(outputs[0]).T 
    
    # Filter by confidence (Deliverable: High Precision)
    conf_threshold = 0.5
    for pred in predictions:
        # Score is the max of the class probabilities
        score = np.max(pred[4:])
        if score > conf_threshold:
            # Scale boxes back to original image size
            cx, cy, bw, bh = pred[:4]
            x1 = int((cx - bw/2) * (w / 416))
            y1 = int((cy - bh/2) * (h / 416))
            x2 = int((cx + bw/2) * (w / 416))
            y2 = int((cy + bh/2) * (h / 416))
            
            cv2.rectangle(orig_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(orig_frame, f"Anomaly {score:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save output
    cv2.imwrite(os.path.join(output_folder, img_name), orig_frame)
    print(f"Processed: {img_name}")

print(f"Done! Check the '{output_folder}' folder.")