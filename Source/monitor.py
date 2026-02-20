"""
Performance monitoring utility for RPi5 Object Detection System
Tracks CPU, memory, temperature, and inference time
"""
import time
import psutil
import os
from typing import Dict, List, Optional
from collections import deque
import threading
class PerformanceMonitor:
    def __init__(self, window_size: int = 30):
        """
        Initializig performance monitor
        Args:
            window_size: Number of samples to keep for averaging
        """
        self.window_size = window_size
        
        # Metric buffers
        self.fps_buffer = deque(maxlen=window_size)
        self.inference_time_buffer = deque(maxlen=window_size)
        self.cpu_buffer = deque(maxlen=window_size)
        self.memory_buffer = deque(maxlen=window_size)
        
        # Counters
        self.total_frames = 0
        self.total_detections = 0
        self.start_time = time.time()
        
        # Temperature monitoring (RPi specific)
        self.temp_file = "/sys/class/thermal/thermal_zone0/temp"
        self.has_temp_sensor = os.path.exists(self.temp_file)
        print(" Performance monitor initialized")
    
    def update_fps(self, fps: float):
        """Updating FPS metric"""
        self.fps_buffer.append(fps)
    
    def update_inference_time(self, inference_time_ms: float):
        """Updating inference time metric"""
        self.inference_time_buffer.append(inference_time_ms)
    
    def update_system_metrics(self):
        """Updating CPU and memory metrics"""
        self.cpu_buffer.append(psutil.cpu_percent(interval=0.1))
        self.memory_buffer.append(psutil.virtual_memory().percent)
    
    def get_temperature(self) -> Optional[float]:
        """
        To Get CPU temperature
        Returns:
            Temperature in Celsius or None if not available
        """
        if not self.has_temp_sensor:
            return None
        
        try:
            with open(self.temp_file, 'r') as f:
                temp = float(f.read().strip()) / 1000.0
            return temp
        except:
            return None
    
    def get_throttled_status(self) -> Optional[str]:
        """
        Checking if RPi is throttled
        Returns:
            Throttling status string or None
        """
        try:
            import subprocess
            result = subprocess.run(
                ['vcgencmd', 'get_throttled'],
                capture_output=True,
                text=True,
                timeout=1
            )
            
            if result.returncode == 0:
                throttled = result.stdout.strip()
                # Parsing throttled value
                if 'throttled=0x0' in throttled:
                    return "OK"
                else:
                    return "THROTTLED"
            return None
        except:
            return None
    
    def get_stats(self) -> Dict[str, float]:
        """
        To Get current statistics
        Returns:
            Dictionary of performance metrics
        """
        stats = {}
        
        # FPS
        if self.fps_buffer:
            stats['fps_current'] = self.fps_buffer[-1]
            stats['fps_avg'] = sum(self.fps_buffer) / len(self.fps_buffer)
            stats['fps_min'] = min(self.fps_buffer)
            stats['fps_max'] = max(self.fps_buffer)
        
        # Inference time
        if self.inference_time_buffer:
            stats['inference_ms_avg'] = sum(self.inference_time_buffer) / len(self.inference_time_buffer)
            stats['inference_ms_min'] = min(self.inference_time_buffer)
            stats['inference_ms_max'] = max(self.inference_time_buffer)
        
        # CPU
        if self.cpu_buffer:
            stats['cpu_avg'] = sum(self.cpu_buffer) / len(self.cpu_buffer)
            stats['cpu_current'] = self.cpu_buffer[-1]
        
        # Memory
        if self.memory_buffer:
            stats['memory_avg'] = sum(self.memory_buffer) / len(self.memory_buffer)
            stats['memory_current'] = self.memory_buffer[-1]
        
        # Temperature
        temp = self.get_temperature()
        if temp:
            stats['temperature_c'] = temp
        
        # Throttling
        throttled = self.get_throttled_status()
        if throttled:
            stats['throttled'] = throttled
        
        # Overall
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            stats['overall_fps'] = self.total_frames / elapsed
        
        stats['total_frames'] = self.total_frames
        stats['total_detections'] = self.total_detections
        stats['elapsed_time'] = elapsed
        
        return stats
    
    def print_stats(self):
        """Print current statistics to console"""
        stats = self.get_stats()
        
        print("\n" + "=" * 60)
        print("PERFORMANCE STATISTICS")
        print("=" * 60)
        
        # FPS
        if 'fps_avg' in stats:
            print(f"FPS: {stats['fps_current']:.1f} "
                  f"(avg: {stats['fps_avg']:.1f}, "
                  f"min: {stats['fps_min']:.1f}, "
                  f"max: {stats['fps_max']:.1f})")
        
        # Inference time
        if 'inference_ms_avg' in stats:
            print(f"Inference: {stats['inference_ms_avg']:.1f}ms "
                  f"(min: {stats['inference_ms_min']:.1f}ms, "
                  f"max: {stats['inference_ms_max']:.1f}ms)")
        
        # CPU and Memory
        if 'cpu_avg' in stats:
            print(f"CPU: {stats['cpu_current']:.1f}% "
                  f"(avg: {stats['cpu_avg']:.1f}%)")
        
        if 'memory_avg' in stats:
            print(f"Memory: {stats['memory_current']:.1f}% "
                  f"(avg: {stats['memory_avg']:.1f}%)")
        
        # Temperature
        if 'temperature_c' in stats:
            temp = stats['temperature_c']
            temp_status = "OK" if temp < 70 else "WARM" if temp < 80 else "HOT"
            print(f"Temperature: {temp:.1f}°C [{temp_status}]")
        
        # Throttling
        if 'throttled' in stats:
            print(f"Throttle Status: {stats['throttled']}")
        
        # Overall
        print(f"Total Frames: {stats['total_frames']}")
        print(f"Total Detections: {stats['total_detections']}")
        print(f"Elapsed Time: {stats['elapsed_time']:.1f}s")
        
        if 'overall_fps' in stats:
            print(f"Overall FPS: {stats['overall_fps']:.2f}")
        
        print("=" * 60 + "\n")
    
    def increment_frames(self):
        """Increment frame counter"""
        self.total_frames += 1
    
    def increment_detections(self, count: int):
        """Increment detection counter"""
        self.total_detections += count
    
    def save_report(self, filepath: str = "performance_report.txt"):
        """
        Save performance report to file
        Args:
            filepath: Path to save report
        """
        stats = self.get_stats()
        
        with open(filepath, 'w') as f:
            f.write("RPi5 Object Detection Performance Report\n")
            f.write("=" * 60 + "\n\n")
            
            for key, value in stats.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.2f}\n")
                else:
                    f.write(f"{key}: {value}\n")
            
            f.write("\n" + "=" * 60 + "\n")
        
        print(f" Performance report saved: {filepath}")


class BackgroundMonitor:
    """
    Background thread for continuous monitoring
    """
    
    def __init__(self, monitor: PerformanceMonitor, interval: float = 1.0):
        """
        Initialize background monitor
        Args:
            monitor: PerformanceMonitor instance
            interval: Update interval in seconds
        """
        self.monitor = monitor
        self.interval = interval
        self.running = False
        self.thread = None
    
    def start(self):
        """Start background monitoring"""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print("Background monitoring started")
    
    def stop(self):
        """Stop background monitoring"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        print("Background monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.running:
            self.monitor.update_system_metrics()
            time.sleep(self.interval)
