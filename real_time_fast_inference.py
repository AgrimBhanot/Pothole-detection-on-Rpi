import cv2
import onnxruntime as ort
import numpy as np
import threading
import time

# ==========================
# SETTINGS
# ==========================
MODEL_PATH = "share/best_preprocessed_excluded.onnx"
IMG_SIZE = 416

# ==========================
# GLOBAL FRAME (LATEST ONLY)
# ==========================
latest_frame = None
lock = threading.Lock()

# ==========================
# CAMERA THREAD
# ==========================
def camera_capture():
    global latest_frame

    cap = cv2.VideoCapture(0)

    # ----- LATENCY CRITICAL SETTINGS -----
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # HUGE latency reducer
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # overwrite previous frame (NO QUEUE)
        with lock:
            latest_frame = frame


# ==========================
# LOAD MODEL
# ==========================
providers = ['CPUExecutionProvider']
session = ort.InferenceSession(MODEL_PATH, providers=providers)

input_name = session.get_inputs()[0].name

# ==========================
# PREPROCESS (FAST)
# ==========================
def preprocess(frame):

    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE),
                       interpolation=cv2.INTER_LINEAR)

    img = frame.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img


# ==========================
# MAIN LOOP
# ==========================
def main():

    global latest_frame

    # start camera thread
    threading.Thread(target=camera_capture, daemon=True).start()

    fps_time = time.time()

    while True:

        with lock:
            frame = latest_frame

        if frame is None:
            continue

        inp = preprocess(frame)

        # ===== INFERENCE =====
        outputs = session.run(None, {input_name: inp})

        # FPS display
        current = time.time()
        fps = 1.0 / (current - fps_time)
        fps_time = current

        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)

        cv2.imshow("Low Latency Feed", frame)

        if cv2.waitKey(1) == 27:
            break


if __name__ == "__main__":
    main()