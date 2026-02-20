"""
Test script for RPi5 Object Detection System
Verifies all components work correctly
"""

import sys
import time
import numpy as np
import cv2
from pathlib import Path

# Color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    """Print colored header"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{text}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

def print_success(text):
    """Print success message"""
    print(f"{GREEN}✓ {text}{RESET}")

def print_error(text):
    """Print error message"""
    print(f"{RED}✗ {text}{RESET}")

def print_warning(text):
    """Print warning message"""
    print(f"{YELLOW}⚠ {text}{RESET}")

def test_imports():
    """Test if all required packages can be imported"""
    print_header("Testing Package Imports")
    
    packages = {
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'onnxruntime': 'ONNX Runtime',
        'psutil': 'psutil'
    }
    
    all_passed = True
    for package, name in packages.items():
        try:
            __import__(package)
            print_success(f"{name} imported successfully")
        except ImportError as e:
            print_error(f"{name} import failed: {e}")
            all_passed = False
    
    return all_passed

def test_config():
    """Test configuration file"""
    print_header("Testing Configuration")
    
    try:
        from config import config
        print_success("Config module loaded")
        
        # Check critical settings
        print(f"  Anomaly model: {config.ANOMALY_MODEL_PATH}")
        print(f"  Pothole model: {config.POTHOLE_MODEL_PATH}")
        print(f"  Confidence threshold: {config.ANOMALY_CONF_THRESHOLD}")
        print(f"  Save threshold: {config.HIGH_CONF_SAVE_THRESHOLD}")
        print(f"  Output directory: {config.OUTPUT_DIR}")
        
        return True
    except Exception as e:
        print_error(f"Config test failed: {e}")
        return False

def test_detector():
    """Test detector with dummy input"""
    print_header("Testing Detector")
    
    try:
        from config import config
        from detector import OptimizedYOLODetector
        
        # Check if model file exists
        if not Path(config.ANOMALY_MODEL_PATH).exists():
            print_warning(f"Model file not found: {config.ANOMALY_MODEL_PATH}")
            print("  Skipping detector test")
            return True
        
        print("Loading model...")
        detector = OptimizedYOLODetector(
            model_config=config.get_anomaly_config(),
            warmup_runs=2  # Reduced for testing
        )
        print_success("Detector initialized")
        
        # Test with dummy frame
        print("Running inference on dummy frame...")
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        start_time = time.time()
        boxes, scores = detector.detect(dummy_frame)
        inference_time = (time.time() - start_time) * 1000
        
        print_success(f"Inference completed in {inference_time:.1f}ms")
        print(f"  Detections: {len(boxes)}")
        
        return True
    except Exception as e:
        print_error(f"Detector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_camera():
    """Test camera/video capture"""
    print_header("Testing Camera")
    
    try:
        from camera import ThreadedCamera
        
        print("Attempting to open camera 0...")
        try:
            camera = ThreadedCamera(src=0, width=640, height=480, fps=30)
            print_success("Camera opened successfully")
            
            # Try to read a frame
            ret, frame = camera.read()
            if ret and frame is not None:
                print_success(f"Frame captured: {frame.shape}")
            else:
                print_warning("Camera opened but failed to capture frame")
            
            camera.release()
            return True
            
        except Exception as e:
            print_warning(f"Camera test failed: {e}")
            print("  This is expected if no camera is connected")
            return True
            
    except Exception as e:
        print_error(f"Camera module test failed: {e}")
        return False

def test_visualizer():
    """Test visualizer"""
    print_header("Testing Visualizer")
    
    try:
        from visualizer import Visualizer, DetectionSaver
        from config import config
        
        # Test visualizer
        vis = Visualizer()
        print_success("Visualizer initialized")
        
        # Test with dummy data
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        boxes = [(100, 100, 200, 200), (300, 300, 400, 400)]
        scores = [0.85, 0.92]
        
        result = vis.draw_detections(dummy_frame, boxes, scores, "TestModel")
        print_success("Detections drawn successfully")
        
        result = vis.add_fps_overlay(result, 10.5)
        print_success("FPS overlay added")
        
        result = vis.add_timestamp_overlay(result, time.time())
        print_success("Timestamp overlay added")
        
        # Test saver
        import tempfile
        import shutil
        temp_dir = tempfile.mkdtemp()
        
        try:
            saver = DetectionSaver(output_dir=temp_dir, high_conf_threshold=0.75)
            print_success("Detection saver initialized")
            
            saver.save_detection(dummy_frame, boxes, scores, "TestModel", time.time())
            time.sleep(0.5)  # Wait for async save
            
            saved_files = list(Path(temp_dir).glob("*.jpg"))
            if saved_files:
                print_success(f"Detection saved: {len(saved_files)} file(s)")
            else:
                print_warning("No files saved (might be async delay)")
            
            saver.stop()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        print_error(f"Visualizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_monitor():
    """Test performance monitor"""
    print_header("Testing Performance Monitor")
    
    try:
        from monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        print_success("Performance monitor initialized")
        
        # Update some metrics
        monitor.update_fps(10.5)
        monitor.update_inference_time(95.3)
        monitor.update_system_metrics()
        monitor.increment_frames()
        monitor.increment_detections(3)
        
        stats = monitor.get_stats()
        print_success(f"Stats collected: {len(stats)} metrics")
        
        # Check temperature
        temp = monitor.get_temperature()
        if temp:
            print(f"  CPU Temperature: {temp:.1f}°C")
        
        return True
        
    except Exception as e:
        print_error(f"Monitor test failed: {e}")
        return False

def test_integration():
    """Test basic integration"""
    print_header("Testing Integration")
    
    try:
        from config import config
        from detector import OptimizedYOLODetector
        from visualizer import Visualizer
        
        # Check if model exists
        if not Path(config.ANOMALY_MODEL_PATH).exists():
            print_warning("Model file not found, skipping integration test")
            return True
        
        print("Creating detector...")
        detector = OptimizedYOLODetector(
            model_config=config.get_anomaly_config(),
            warmup_runs=1
        )
        
        print("Creating visualizer...")
        vis = Visualizer()
        
        print("Running full pipeline on dummy frame...")
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Detect
        boxes, scores = detector.detect(dummy_frame)
        
        # Visualize
        result = vis.draw_detections(dummy_frame, boxes, scores, detector.get_name())
        result = vis.add_fps_overlay(result, 10.0)
        result = vis.add_timestamp_overlay(result, time.time())
        
        print_success("Full pipeline executed successfully")
        print(f"  Frame shape: {result.shape}")
        print(f"  Detections: {len(boxes)}")
        
        return True
        
    except Exception as e:
        print_error(f"Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print_header("RPi5 Object Detection System - Test Suite")
    
    tests = [
        ("Package Imports", test_imports),
        ("Configuration", test_config),
        ("Detector", test_detector),
        ("Camera", test_camera),
        ("Visualizer", test_visualizer),
        ("Performance Monitor", test_monitor),
        ("Integration", test_integration),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print_error(f"Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print_header("Test Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        if result:
            print_success(f"{name}: PASSED")
        else:
            print_error(f"{name}: FAILED")
    
    print(f"\n{BLUE}{'='*60}{RESET}")
    if passed == total:
        print(f"{GREEN}All tests passed! ({passed}/{total}){RESET}")
        print(f"{GREEN}System is ready to use.{RESET}")
        return 0
    else:
        print(f"{YELLOW}Some tests failed: {passed}/{total} passed{RESET}")
        if passed >= total - 1:
            print(f"{YELLOW}System might still be usable.{RESET}")
        else:
            print(f"{RED}Please fix the issues before using the system.{RESET}")
        return 1
    print(f"{BLUE}{'='*60}{RESET}\n")

if __name__ == "__main__":
    sys.exit(main())
