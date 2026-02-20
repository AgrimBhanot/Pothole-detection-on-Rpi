# RPi5 Object Detection System

**High-performance object detection optimized for Raspberry Pi 5**

Dual-model system with multiprocessing, zero-copy operations, and advanced optimizations for real-time inference.

---

## 🚀 Features

- **Dual Model Support**: Alternates between Anomaly Detection and Pothole Detection models
- **Optimized for RPi5**: 
  - Vectorized NumPy operations
  - Zero-copy preprocessing
  - NMS for duplicate removal
  - ONNX Runtime optimizations
  - Model warmup for consistent performance
- **Threaded Camera Capture**: Prevents buffer lag
- **Async Detection Saving**: Background thread for high-confidence detections saving WITH TIME-STAMPS (>0.75)
- **Real-time Visualization**:
  - Live FPS display
  - Timestamp overlay
  - Detection count
  - Model name indicator
- **Smart Frame Management**: Drops old frames to maintain low latency

---

## 📁 Project Structure

```
.
├── new_model/                #Saves all the models
│   ├── best_preprocessed.onnx
│   ├── best.onnx
│   ├── best.pt
│   ├── Model1_Pothole.onnx    #Main Pothole model in use
│   └── Model2_General.onnx    #General Model in use
│
├── Source/
│   ├── __pycache__/
│   ├── camera.py              # Threaded camera capture
│   ├── config.py              # Centralized configuration
│   ├── detector.py            # Optimized YOLO detector class
│   ├── main.py                # Main application
│   ├── monitor.py             # System/resource monitoring
│   ├── pipeline.py            # Multiprocessing pipeline (advanced)
│   ├── test_system.py         # System testing utilities
│   └── visualizer.py          # Drawing and saving utilities
│
├── calibrate_and_excluded_Quantization.py # Quantization to INT8 with exculded layers
├── image_testing.py           # Image-based testing script
├── Preprocessing.py           # Model preprocessing before Quantization
├── .gitignore
├── README.md
├── requirements.txt           # Python dependencies
└── setup.sh                   # Setup script
```

---

## 🔧 Installation

### 1. System Prerequisites (RPi5)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install OpenCV dependencies
sudo apt install -y python3-opencv libopencv-dev

# Install numpy (optimized for ARM)
sudo apt install -y python3-numpy

# Install pip
sudo apt install -y python3-pip
```

### 2. Python Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Model Files

Place your ONNX model files in the project directory:
- `Model2_General.onnx` (Anomaly detection model)
- `Model1_Pothole.onnx` (Pothole detection model)

Update paths in `Source/config.py` if needed.

### 4. Alternative
Or just run the setup.sh script.
---

## 🎮 Usage

### Basic Commands

**Run with camera (dual models):**
```bash
python Source/main.py --source camera
```

**Run with camera (single model):**
```bash
python Source/main.py --source camera --single-model
```

**Run with video file:**
```bash
python Source/main.py --source path/to/video.mp4
```

**Disable automatic saving:**
```bash
python Source/main.py --source camera --no-save
```

**Custom thresholds:**
```bash
python Source/main.py --source camera --conf-threshold 0.6 --save-threshold 0.8
```

### Keyboard Controls

- **`q`** - Quit application
- **`p`** - Pause/Resume detection
- **`s`** - Manually save current frame

---

## ⚙️ Configuration

Edit `Source/config.py` to customize:

### Model Settings
```python
ANOMALY_MODEL_PATH = "new_model/Model2_General.onnx"
POTHOLE_MODEL_PATH = "new_model/Model1_Pothole.onnx"
ANOMALY_CONF_THRESHOLD = 0.7
POTHOLE_CONF_THRESHOLD = 0.7
```

```

### Output Settings
```python
SAVE_DETECTIONS = True
OUTPUT_DIR = "detections"
DISPLAY_FPS = True
SHOW_TIMESTAMPS = True
```

---

## 🔥 Performance Optimization Tips

### 1. Overclock RPi5 (Optional)

⚠️ **Requires active cooling!**

Edit `/boot/firmware/config.txt`:
```ini
arm_freq=2800
over_voltage_delta=50000
```

Reboot and verify:
```bash
vcgencmd measure_clock arm
```

### 2. Monitor Thermals

```bash
# Check temperature
vcgencmd measure_temp

# Monitor in real-time
watch -n 1 vcgencmd measure_temp
```

Keep temperature below 80°C for optimal performance.

### 3. Reduce GUI Load

For maximum performance, run without desktop environment:
```bash
# Disable desktop
sudo systemctl set-default multi-user.target
sudo reboot

# Re-enable later if needed
sudo systemctl set-default graphical.target
```

### 4. Process Priority

Run with higher priority:
```bash
sudo nice -n -10 python main.py --source camera
```

---

## 📊 Performance Benchmarks

Expected performance on RPi5 (with active cooling):

| Configuration | FPS | CPU Usage | Notes |
|--------------|-----|-----------|-------|
| Single Model (416x416) | 8-10 | ~70% | Stable |
| Dual Alternating (416x416) | 8-10 | ~75% | Slight overhead |
| With Display | 8-10 | +5% | OpenCV imshow |
| Overclocked (2.8GHz) | 10-12 | ~70% | Requires cooling |

---

## 📝 Output

### Saved Detections

High-confidence detections (>0.75) are automatically saved to `detections/` with format:
```
ModelName_YYYYMMDD_HHMMSS_microseconds_confidence.jpg
```

Example:
```
Anomaly_20260220_143052_123456_0.89.jpg
Pothole_20260220_143053_789012_0.92.jpg
```

### Console Output

```
FPS: 9.2 | Frame: 1234 | Detections: 3 | Saved: 45
```

---

## 🐛 Troubleshooting

### Low FPS (<5 FPS)

1. Check CPU throttling:
   ```bash
   vcgencmd get_throttled
   ```
   If throttled, improve cooling or reduce overclock.

2. Reduce input resolution in `config.py`:
   ```python
   CAMERA_WIDTH = 320
   CAMERA_HEIGHT = 240
   ```

3. Use single model mode:
   ```bash
   python main.py --source camera --single-model
   ```

### Camera Not Found

```bash
# List available cameras
v4l2-ctl --list-devices

# Test camera
raspistill -o test.jpg
```

### ONNX Runtime Errors

Ensure you have the correct ONNX Runtime:
```bash
pip install onnxruntime==1.16.3
```

For ARM optimization:
```bash
pip uninstall onnxruntime
pip install onnxruntime  # Gets ARM-optimized version
```

### Memory Issues

Monitor memory usage:
```bash
free -h
```

If low on memory:
1. Reduce queue size in `config.py`:
   ```python
   FRAME_QUEUE_SIZE = 1
   ```

2. Disable detection saving:
   ```bash
   python main.py --source camera --no-save
   ```

---

## 🔬 Advanced: True Multiprocessing

The current implementation uses threading for simplicity. For true multiprocessing (separate CPU cores), you need to:

1. Serialize/deserialize model data
2. Use `multiprocessing.Queue` for frame passing
3. Load models in separate processes

See `pipeline.py` for the framework. This requires more complex IPC but can achieve better CPU utilization.

---

## 📈 Monitoring Performance

### Real-time Stats

Add to main loop in `main.py`:
```python
import psutil

cpu_percent = psutil.cpu_percent(interval=1)
memory_info = psutil.virtual_memory()
print(f"CPU: {cpu_percent}% | RAM: {memory_info.percent}%")
```

### Profiling

```bash
# Install profiler
pip install py-spy

# Profile running script
sudo py-spy top --pid <PID>

# Or run with profiler
sudo py-spy record -o profile.svg -- python main.py --source camera
```

---

## 🎯 Deliverables Checklist

- ✅ **FPS displayed on screen** - Real-time in top-left corner
- ✅ **High-confidence image saving** - Automatic save when conf > 0.75
- ✅ **Timestamp on saved images** - Filename includes full timestamp
- ✅ **Bounding boxes drawn** - Color-coded by model type
- ✅ **Dual model support** - Alternates to maintain performance
- ✅ **Optimized for RPi5** - Vectorized ops, NMS, zero-copy, warmup
- ✅ **Low latency** - Frame dropping, small queues, threaded capture

---

## 📄 License

MIT License - Feel free to use and modify.

---

**Built for RPi5 with performace and optimization in mind**
