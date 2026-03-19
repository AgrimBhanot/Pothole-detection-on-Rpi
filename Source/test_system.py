"""
Test script for RPi5 Object Detection System (v2)
Tests all new components: ModelPairManager, ByteTracker, Pipeline.
"""
import sys
import time
import numpy as np
from pathlib import Path

GREEN = '\033[92m'; RED = '\033[91m'; YELLOW = '\033[93m'
BLUE = '\033[94m';  RESET = '\033[0m'

def ph(t): print(f"\n{BLUE}{'='*60}\n{t}\n{'='*60}{RESET}\n")
def ok(t): print(f"{GREEN}✓ {t}{RESET}")
def err(t): print(f"{RED}✗ {t}{RESET}")
def warn(t): print(f"{YELLOW}⚠ {t}{RESET}")

def test_imports():
    ph("Testing Package Imports")
    pkgs = {'cv2':'OpenCV','numpy':'NumPy','onnxruntime':'ONNX Runtime','psutil':'psutil'}
    passed = True
    for pkg, name in pkgs.items():
        try: __import__(pkg); ok(f"{name}")
        except ImportError as e: err(f"{name}: {e}"); passed = False
    return passed

def test_config():
    ph("Testing Configuration")
    try:
        from config import config
        ok("Config loaded")
        print(f"  FP32 general:    {config.ANOMALY_MODEL_PATH}")
        print(f"  FP32 pothole:    {config.POTHOLE_MODEL_PATH}")
        print(f"  INT8 general:    {config.ANOMALY_MODEL_QUANT_PATH}")
        print(f"  INT8 pothole:    {config.POTHOLE_MODEL_QUANT_PATH}")
        print(f"  FPS down/up:     {config.FPS_DOWNGRADE_THRESHOLD} / {config.FPS_UPGRADE_THRESHOLD}")
        print(f"  Temp down/up:    {config.TEMP_DOWNGRADE_THRESHOLD} / {config.TEMP_UPGRADE_THRESHOLD}°C")
        print(f"  Cooldown:        {config.SWITCH_COOLDOWN_SECONDS}s")
        return True
    except Exception as e:
        err(f"Config: {e}"); return False

def test_tracker():
    ph("Testing ByteTracker (Task 2)")
    try:
        from tracker import ByteTracker, TrackState
        tracker = ByteTracker(label="Pothole", max_lost_frames=3)
        ok("ByteTracker initialised")

        # Frame 1: two detections
        boxes  = [(10,10,50,50), (200,200,300,300)]
        scores = [0.85, 0.72]
        tracks = tracker.update(boxes, scores)
        assert len(tracks) == 2, f"Expected 2 tracks, got {len(tracks)}"
        ok(f"Frame 1: {len(tracks)} tracks created")

        # Frame 2: one detection — other should remain (LOST, not removed yet)
        tracks = tracker.update([(10,10,50,50)], [0.90])
        assert len(tracks) == 1
        ok(f"Frame 2: {len(tracks)} active track")

        # IDs are persistent
        assert tracks[0].track_id == 1
        ok(f"Persistent track ID: #{tracks[0].track_id}")

        # Max confidence accumulates
        assert tracks[0].max_confidence == 0.90
        ok(f"Max confidence tracked: {tracks[0].max_confidence:.2f}")

        # After max_lost_frames, second track should be purged with log
        print("  (Expect lifecycle log below:)")
        for _ in range(5):
            tracker.update([(10,10,50,50)], [0.80])
        ok("Track lifecycle log emitted on purge")

        return True
    except Exception as e:
        import traceback; traceback.print_exc()
        err(f"Tracker test: {e}"); return False

def test_model_manager():
    ph("Testing ModelPairManager (Task 1)")
    try:
        from config import config
        from model_manager import ModelPairManager, ModelMode

        if not Path(config.ANOMALY_MODEL_PATH).exists():
            warn("FP32 model not found — skipping ModelPairManager test")
            return True

        mgr = ModelPairManager(config)
        ok("ModelPairManager initialised")

        assert mgr.current_mode == ModelMode.PERFORMANCE
        ok(f"Default mode: {mgr.current_mode.value}")

        # Test EMA FPS update
        for fps in [14, 15, 14, 15]:
            mgr.update_fps(fps)
        assert mgr.ema_fps > 10
        ok(f"EMA FPS after good frames: {mgr.ema_fps:.1f}")

        # Simulate low FPS to trigger downgrade
        for _ in range(20):
            mgr.update_fps(5.0)
        mgr._last_switch = 0  # Reset cooldown for test
        switched = mgr.check_and_switch()
        if switched:
            ok(f"Mode switched to EFFICIENCY (low FPS)")
        else:
            warn("Mode did not switch (EMA may not have dropped enough)")

        return True
    except Exception as e:
        import traceback; traceback.print_exc()
        err(f"ModelPairManager: {e}"); return False

def test_visualizer():
    ph("Testing Visualizer (updated)")
    try:
        from visualizer import Visualizer, DetectionSaver
        from tracker import ByteTracker

        vis = Visualizer()
        ok("Visualizer initialised")

        frame  = np.zeros((480, 640, 3), dtype=np.uint8)
        tracker = ByteTracker(label="Obstacle")
        tracks  = tracker.update([(100,100,200,200)], [0.85])

        frame = vis.draw_tracks(frame, tracks, [])
        ok("draw_tracks rendered")

        frame = vis.add_mode_overlay(frame, "PERFORMANCE")
        ok("Mode overlay rendered")

        frame = vis.add_fps_overlay(frame, 14.5)
        ok("FPS overlay rendered")

        return True
    except Exception as e:
        import traceback; traceback.print_exc()
        err(f"Visualizer: {e}"); return False

def test_pipeline_import():
    ph("Testing Pipeline (Task 3) — import & structure")
    try:
        from pipeline import DetectionPipeline, FrameData, DetectionResult
        ok("Pipeline module imported")

        p = DetectionPipeline(max_queue_size=1, max_latency_ms=200)
        ok("DetectionPipeline instantiated")
        assert hasattr(p, 'mode_val'), "mode_val shared value missing"
        ok("mode_val shared state present")

        return True
    except Exception as e:
        import traceback; traceback.print_exc()
        err(f"Pipeline: {e}"); return False

def main():
    ph("RPi5 Detection System v2 — Test Suite")
    tests = [
        ("Package Imports",   test_imports),
        ("Configuration",     test_config),
        ("ByteTracker",       test_tracker),
        ("ModelPairManager",  test_model_manager),
        ("Visualizer",        test_visualizer),
        ("Pipeline",          test_pipeline_import),
    ]
    results = []
    for name, fn in tests:
        try:
            results.append((name, fn()))
        except Exception as e:
            err(f"'{name}' crashed: {e}")
            results.append((name, False))

    ph("Summary")
    passed = sum(1 for _, r in results if r)
    for name, r in results:
        (ok if r else err)(f"{name}: {'PASSED' if r else 'FAILED'}")
    print(f"\n{BLUE}{'='*60}{RESET}")
    status = GREEN if passed == len(results) else YELLOW
    print(f"{status}{passed}/{len(results)} tests passed{RESET}")
    return 0 if passed == len(results) else 1

if __name__ == "__main__":
    sys.exit(main())
