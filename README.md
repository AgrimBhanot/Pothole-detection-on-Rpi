# RPi5 Real-Time Object Detection System

**Production-grade dual-model object detection pipeline, optimized for Raspberry Pi 5.**

Runs two YOLO models (general anomaly + pothole) in a true multiprocessing architecture with adaptive model-pair switching, ByteTrack object tracking, hardware-accelerated NMS, and a full performance monitoring stack.

---

## What Was Built — v2 Upgrades at a Glance

| Feature | Details |
|---------|---------|
| **True multiprocessing pipeline** | 3 separate OS processes, each on its own CPU cores — bypasses Python GIL |
| **Adaptive hysteresis model switching** | Auto-switches between FP32 and INT8 model pairs based on EMA FPS + CPU temperature |
| **ByteTrack object tracking** | Persistent object IDs across frames, two-stage IoU matching, re-identification |
| **OpenCV hardware-accelerated NMS** | `cv2.dnn.NMSBoxes()` — C++ vectorized, replaces Python NMS loop |
| **Lazy model loading** | Only the active model pair is in RAM — halves memory pressure |
| **Drop-frame queuing** | `mp.Queue(maxsize=1)` with drop-old strategy — zero backlog, always latest frame |
| **Latency guard** | Frames older than 200 ms are automatically discarded |
| **EMA FPS smoothing** | α=0.1 exponential moving average — immune to single-frame GIL pauses |
| **Thermal management** | Live sysfs temperature read, `vcgencmd` throttle detection |
| **INT8 quantization tooling** | Calibration script with per-layer exclusion control |

---

## Features

### Core Detection
- **Dual YOLO model support** — alternates between general anomaly and pothole detector every frame
- **Confidence thresholding** — configurable per model (default 0.70)
- **High-confidence saving** — detections above 0.75 saved to disk with full timestamps
- **Bounding boxes** — color-coded by model type (green = anomaly, red = pothole)

### Multiprocessing Pipeline
- **3 OS processes** — `CaptureProcess`, `InferenceProcess`, `MainProcess` run in true parallel on separate CPU cores
- **`mp.Queue(maxsize=1)`** — each inter-process queue holds exactly 1 item; producer drops the old item and inserts the new one on overflow, so the consumer always sees the latest data and never builds a backlog
- **`mp.Value`** — shared memory integers for stop signal (`running`) and current model mode (`mode_val`); zero-copy, no serialization, instant read
- **Module-level worker functions** — `_capture_worker` and `_inference_worker` defined at module level so they are picklable for `mp.Process`
- **Daemon processes** — both child processes are `daemon=True`; they die automatically if the main process exits

### Adaptive Model-Pair Switching
- **Two model pairs** — PERFORMANCE (FP32, full precision) and EFFICIENCY (INT8 quantized)
- **Lazy loading** — only the active pair's ONNX sessions are in RAM; switching frees old sessions before loading new ones
- **EMA FPS smoothing** — `α = 0.1`; each new sample contributes 10% to the running average, preventing single slow frames from triggering a switch
- **Hysteresis dead-band** — separate downgrade (FPS < 4.0 or temp > 75 °C) and upgrade (FPS > 7.0 and temp < 65 °C) thresholds; the 3 FPS and 10 °C gaps prevent oscillation
- **Cooldown** — 10-second minimum between any two successive switches using `time.monotonic()`
- **Graceful degradation** — if INT8 weights are absent, efficiency mode silently reuses the FP32 pair

### ByteTrack Object Tracking
- **Persistent IDs** — every detected object gets a unique integer ID that persists across frames
- **Two-stage matching** — Stage 1: high-confidence detections (≥ 0.60) matched to active tracks by IoU; Stage 2: low-confidence detections (0.30–0.60) matched to remaining active tracks; Stage 3: re-identification against lost tracks; Stage 4: new track birth
- **Re-identification** — tracks held for 15 frames after going lost before purge, allowing objects to re-appear
- **Lifecycle logging** — on purge: `[Tracker] Encountered Pothole #3 | Duration: 4.2s | Max Confidence: 0.87`
- **Two independent trackers** — one for the general model, one for the pothole model, routed by `is_pothole_frame` flag

### Inference Optimizations
- **ONNX Runtime tuning** — `ORT_ENABLE_ALL` graph optimization, `ORT_SEQUENTIAL` execution mode, `intra_op_num_threads=2`, `inter_op_num_threads=1`
- **Thread affinity** — ONNX intra-op threads occupy Cores 1+2; capture on Core 0; display on Core 3; no process fights another for cores
- **Model warmup** — 10 dummy inferences on session creation to warm CPU instruction and data caches before live use
- **Alternating model schedule** — frame N runs general detector, frame N+1 runs pothole detector; one ONNX session per frame, halving per-frame compute
- **Vectorized pre/postprocessing** — NumPy operations throughout; zero-copy input buffer allocation
- **`cv2.dnn.NMSBoxes()`** — hardware-accelerated C++ NMS with `conf_threshold=0.70`, `nms_threshold=0.45`

### Visualization
- **`draw_tracks()`** — renders bounding boxes with persistent track IDs and confidence scores
- **`add_mode_overlay()`** — PERFORMANCE (cyan) / EFFICIENCY (yellow) badge in bottom-right corner
- **FPS overlay** — live FPS in top-left
- **Timestamp overlay** — wall-clock time on every frame
- **Detection count** — total detections this frame
- **Model name indicator** — shows which model ran on the current frame
- **High-confidence save** — `DetectionSaver` runs on a background thread; saves `ModelName_YYYYMMDD_HHMMSS_microseconds_confidence.jpg`

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Raspberry Pi 5  (4× Cortex-A76)                  │
│                                                                         │
│  Core 0                 Core 1 + 2               Core 3                 │
│  ┌──────────────┐       ┌──────────────────┐     ┌──────────────────┐   │
│  │CaptureProcess│       │InferenceProcess  │     │MainProcess       │   │
│  │              │       │                  │     │                  │   │
│  │ ThreadedCam  │──Q1──▶│ ModelPairManager │─Q2─▶│ ByteTracker ×2   │   │
│  │ FrameData    │       │ Alternating mods │     │ Visualizer       │   │
│  │ drop-old     │       │ Hysteresis check │     │ DetectionSaver   │   │
│  └──────────────┘       │ EMA FPS update   │     │PerformanceMonitor│   │
│                         └──────────────────┘     └──────────────────┘   │
│                                                                         │
│  Q1, Q2 = mp.Queue(maxsize=1)  — drop-old on overflow                   │
│  mp.Value: running (stop signal) · mode_val (PERF/EFF)                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
.
├── Model_modification/
│   ├── calibrate_and_exculded_Quantization.py  # INT8 calibration with per-layer exclusion
│   ├── image_testing.py                         # Pre/post-quantization accuracy validation
│   └── Preprocessing.py                         # ONNX graph preprocessing before quantization
│
├── new_model/                                   # Model weights 
│   ├── best.onnx                                # FP32 general model (dev/test)
│   ├── best.pt                                  # PyTorch source weights
│   ├── Model1_Pothole.onnx                      # FP32 pothole detector (production)
│   └── Model2_General.onnx                      # FP32 general anomaly detector (production)
│
├── Source/
│   ├── camera.py          # ThreadedCamera — non-blocking capture with daemon thread
│   ├── config.py          # Centralized configuration — all tunable parameters
│   ├── detector.py        # OptimizedYOLODetector — ONNX Runtime + OpenCV NMS
│   ├── main.py            # Entry point — wires pipeline, trackers, visualizer
│   ├── model_manager.py   # ModelPairManager — lazy loading, hysteresis switching
│   ├── monitor.py         # PerformanceMonitor + BackgroundMonitor
│   ├── pipeline.py        # DetectionPipeline — 3-process mp architecture
│   ├── test_system.py     # Unit + integration tests for all components
│   ├── tracker.py         # ByteTracker — two-stage IoU matching, re-ID
│   └── visualizer.py      # draw_tracks, add_mode_overlay, DetectionSaver
│
├── .gitignore
├── README.md
├── requirements.txt
└── setup.sh
```

---

## Installation

### 1. Just run setup

```bash
bash setup.sh
```

---

## Usage

```bash
# Live camera — dual model, full pipeline
python Source/main.py --source camera

# Live camera — single model
python Source/main.py --source camera --single-model

# Video file
python Source/main.py --source path/to/video.mp4

# Disable saving
python Source/main.py --source camera --no-save

# Custom thresholds
python Source/main.py --source camera --conf-threshold 0.6 --save-threshold 0.8
```

### Keyboard controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `p` | Pause / Resume |
| `s` | Manually save current frame |

---

## Configuration (`Source/config.py`)

All tunable parameters live here. Nothing is hardcoded elsewhere.

### Model paths
```python
ANOMALY_MODEL_PATH       = "new_model/Model2_General.onnx"
POTHOLE_MODEL_PATH       = "new_model/Model1_Pothole.onnx"
ANOMALY_MODEL_QUANT_PATH = "new_model/Model2_General_quant.onnx"
POTHOLE_MODEL_QUANT_PATH = "new_model/Model1_Pothole_quant.onnx"
```


### ONNX Runtime threads (tuned for Pi5 4-core layout)
```python
INTRA_OP_NUM_THREADS = 2   # Cores used per ONNX session (Cores 1+2)
INTER_OP_NUM_THREADS = 1
NUM_WARMUP_RUNS      = 10  # Dummy inferences to warm caches on session load
```


### ByteTrack settings
```python
TRACKER_MAX_LOST_FRAMES     = 15    # Frames before a lost track is purged
TRACKER_IOU_THRESHOLD       = 0.35  # Min IoU for a valid match
TRACKER_HIGH_CONF_THRESHOLD = 0.60  # High-confidence detection tier
TRACKER_LOW_CONF_THRESHOLD  = 0.30  # Low-confidence detection tier
```

---

## Performance

Measured on RPi5 with active cooling, 640×480 input:

| Configuration | FPS | CPU | Notes |
|---------------|-----|-----|-------|
| Single model FP32 (416×416) | 8–10 | ~70% | Stable |
| Dual alternating FP32 (416×416) | 8–10 | ~75% | One model per frame |
| Dual alternating INT8 (416×416) | 10–14 | ~65% | EFFICIENCY mode |
| Overclocked 2.8 GHz + INT8 | 12–16 | ~65% | Requires active cooling |
| With cv2.imshow | −1 to −2 FPS | +5% | OpenCV GUI overhead |

### Model switch latency
Switching pairs takes 700 ms – 2 s (dominated by 10 warmup runs × inference time). The 10-second cooldown ensures this only happens when genuinely necessary. Reduce `NUM_WARMUP_RUNS` to 2–3 for faster switches at the cost of slightly slower first frames.

---

## Performance Tuning Tips

### Overclock RPi5 (requires active cooling)
```bash
# Edit /boot/firmware/config.txt
arm_freq=2800
over_voltage_delta=50000
```
```bash
vcgencmd measure_clock arm   # Verify
vcgencmd measure_temp        # Monitor
```

### Run headless (maximum throughput)
```bash
sudo systemctl set-default multi-user.target && sudo reboot
# Restore: sudo systemctl set-default graphical.target
```

### Raise process priority
```bash
sudo nice -n -10 python Source/main.py --source camera
```

### Monitor thermals
```bash
watch -n 1 vcgencmd measure_temp
watch -n 1 vcgencmd get_throttled   # 0x0 = no throttle
```
Keep below 80 °C. The system will auto-downgrade to EFFICIENCY mode at 75 °C.

---

## Output

### Saved detections
High-confidence frames (> 0.75) saved automatically to `detections/`:
```
ModelName_YYYYMMDD_HHMMSS_microseconds_confidence.jpg
```
Example:
```
Pothole_20260220_143052_123456_0.92.jpg
Anomaly_20260220_143053_789012_0.87.jpg
```

### Console output
```
[Pipeline] Started — CaptureProcess PID 1234, InferenceProcess PID 1235
[ModelManager] Ready. Default mode: EFFICIENCY | Quantized pair loaded: True
[InferenceWorker] Started.
FPS: 9.2 | Frame: 1234 | Detections: 3 | Saved: 45
[ModelManager] Switched to EFFICIENCY due to: low FPS (3.8 < 4.0)
[Tracker] Encountered Pothole #3 | Duration: 4.2s | Max Confidence: 0.87
```

---

## Troubleshooting

### Low FPS (< 5)
```bash
vcgencmd get_throttled   # Non-zero = clock is throttled — improve cooling
```
- Lower resolution: set `CAMERA_WIDTH=320, CAMERA_HEIGHT=240` in `config.py`
- Use `--single-model` flag
- Check that `INTRA_OP_NUM_THREADS=2` — setting to 4 causes sessions to compete for cores

### Camera not found
```bash
v4l2-ctl --list-devices
raspistill -o test.jpg
```

### ONNX Runtime errors
```bash
pip uninstall onnxruntime && pip install onnxruntime   # Gets ARM-optimized build
```

### Memory pressure
- Set `FRAME_QUEUE_SIZE=1` (already default)
- Run `--no-save` to disable background saving thread
- Use `--single-model` to load one ONNX session instead of two
- Monitor: `free -h` — if < 500 MB free, consider reducing model input size

### Model switch freezes display
Normal — inference is paused during `_load_active_pair()`. Reduce `NUM_WARMUP_RUNS` from 10 to 3 in `config.py` to cut switch time from ~1.5 s to ~400 ms.

---

## Deliverables Checklist

- ✅ FPS displayed on screen — real-time top-left overlay
- ✅ High-confidence image saving — auto-save when conf > 0.75
- ✅ Timestamp on saved images — full datetime in filename
- ✅ Bounding boxes drawn — color-coded by model
- ✅ Dual model support — alternates to maintain performance
- ✅ ByteTrack tracking — persistent IDs, re-ID, lifecycle logging
- ✅ True multiprocessing — 3 OS processes, GIL bypassed
- ✅ Adaptive model switching — EMA FPS + thermal hysteresis
- ✅ INT8 quantization tooling — calibration with layer exclusion
- ✅ Optimized for RPi5 — vectorized ops, NMS, zero-copy, ORT tuning, thread affinity
- ✅ Low latency — drop-frame queues, latency guard, lazy model loading

---

## Dependencies

See `requirements.txt`. Key packages:
- `onnxruntime` — ARM-optimized inference engine
- `opencv-python` — NMS, preprocessing, display
- `numpy` — vectorized pre/postprocessing
- `picamera2` — CSI camera interface
- `psutil` — CPU and memory monitoring

---

## License

MIT License — free to use and modify.

---

*Built for RPi5 with performance, thermal stability, and real-time reliability in mind.*