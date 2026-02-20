import cv2
import numpy as np
import onnxruntime as ort
import argparse
import sys

class YOLOv8Detector:
    def __init__(self, model_path, conf_threshold=0.5):
        """Initialize the YOLO detector with ONNX model"""
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        
        # Setup ONNX Runtime Session
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        
        # Get model input shape
        self.input_width = 416
        self.input_height = 416
        
        print(f"✓ Model loaded: {model_path}")
        print(f"✓ Confidence threshold: {conf_threshold}")
    
    def preprocess(self, frame):
        """Preprocess frame for model inference"""
        # Resize to model input size
        img = cv2.resize(frame, (self.input_width, self.input_height))
        
        # Normalize to [0, 1]
        img_data = img.astype(np.float32) / 255.0
        
        # Transpose to CHW format (channels first)
        img_data = np.transpose(img_data, (2, 0, 1))
        
        # Add batch dimension
        img_data = np.expand_dims(img_data, axis=0)
        
        return img_data
    
    def postprocess(self, outputs, frame):
        """Process model outputs and draw bounding boxes"""
        h, w = frame.shape[:2]
        
        # YOLOv8 output is typically [1, 84, 8400]
        # Transpose to [8400, 84] for easier processing
        predictions = np.squeeze(outputs[0]).T
        
        boxes = []
        scores = []
        
        # Process each prediction
        for pred in predictions:
            # Get class scores (skip first 4 values which are box coordinates)
            class_scores = pred[4:]
            score = np.max(class_scores)
            
            if score > self.conf_threshold:
                # Get box coordinates
                cx, cy, bw, bh = pred[:4]
                
                # Scale boxes back to original image size
                x1 = int((cx - bw/2) * (w / self.input_width))
                y1 = int((cy - bh/2) * (h / self.input_height))
                x2 = int((cx + bw/2) * (w / self.input_width))
                y2 = int((cy + bh/2) * (h / self.input_height))
                
                # Clamp coordinates to image boundaries
                x1 = max(0, min(x1, w))
                y1 = max(0, min(y1, h))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))
                
                boxes.append((x1, y1, x2, y2))
                scores.append(score)
        
        return boxes, scores
    
    def draw_detections(self, frame, boxes, scores):
        """Draw bounding boxes and labels on frame"""
        for (x1, y1, x2, y2), score in zip(boxes, scores):
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label with confidence score
            label = f"Detection {score:.2f}"
            
            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                frame, 
                (x1, y1 - text_height - 10), 
                (x1 + text_width, y1), 
                (0, 255, 0), 
                -1
            )
            
            # Draw text
            cv2.putText(
                frame, 
                label, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 0), 
                2
            )
        
        return frame
    
    def detect(self, frame):
        """Run detection on a single frame"""
        # Preprocess
        img_data = self.preprocess(frame)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: img_data})
        
        # Postprocess
        boxes, scores = self.postprocess(outputs, frame)
        
        # Draw detections
        result_frame = self.draw_detections(frame.copy(), boxes, scores)
        
        return result_frame, len(boxes)


def process_camera(detector, camera_id=0):
    """Process camera feed in real-time"""
    print(f"\n📹 Starting camera feed (ID: {camera_id})")
    print("Press 'q' to quit, 'p' to pause/resume")
    
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open camera {camera_id}")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    paused = False
    frame_count = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Failed to grab frame")
                break
            
            frame_count += 1
            
            # Run detection
            result_frame, num_detections = detector.detect(frame)
            
            # Add FPS and detection info
            info_text = f"Frame: {frame_count} | Detections: {num_detections}"
            cv2.putText(
                result_frame, 
                info_text, 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2
            )
            
            # Display
            cv2.imshow('Real-Time Object Detection - Camera', result_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print(f"{'⏸️  Paused' if paused else '▶️  Resumed'}")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✓ Processed {frame_count} frames")


def process_video(detector, video_path, save_output=False, output_path=None):
    """Process video file"""
    print(f"\n🎬 Processing video: {video_path}")
    print("Press 'q' to quit, 'p' to pause/resume")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Setup video writer if saving output
    writer = None
    if save_output:
        if output_path is None:
            output_path = "output_detection.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"💾 Saving output to: {output_path}")
    
    paused = False
    frame_count = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n✓ End of video reached")
                break
            
            frame_count += 1
            
            # Run detection
            result_frame, num_detections = detector.detect(frame)
            
            # Add progress info
            progress = (frame_count / total_frames) * 100
            info_text = f"Frame: {frame_count}/{total_frames} ({progress:.1f}%) | Detections: {num_detections}"
            cv2.putText(
                result_frame, 
                info_text, 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2
            )
            
            # Save frame if writer is active
            if writer is not None:
                writer.write(result_frame)
            
            # Display
            cv2.imshow('Real-Time Object Detection - Video', result_frame)
            
            # Print progress every 30 frames
            if frame_count % 30 == 0:
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        # Handle key presses (adjust wait time based on pause state)
        key = cv2.waitKey(1 if not paused else 30) & 0xFF
        if key == ord('q'):
            print("\n⏹️  Stopped by user")
            break
        elif key == ord('p'):
            paused = not paused
            print(f"{'⏸️  Paused' if paused else '▶️  Resumed'}")
    
    cap.release()
    if writer is not None:
        writer.release()
        print(f"✓ Output saved to: {output_path}")
    cv2.destroyAllWindows()
    print(f"✓ Processed {frame_count}/{total_frames} frames")


def main():
    parser = argparse.ArgumentParser(description='Real-time Object Detection with YOLOv8 ONNX')
    parser.add_argument('--model', type=str, default='new_model/new_preprocessed_excluded.onnx',
                        help='Path to ONNX model file')
    parser.add_argument('--source', type=str, default='camera',
                        help='Input source: "camera" or path to video file')
    parser.add_argument('--camera-id', type=int, default=0,
                        help='Camera ID (default: 0)')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold (default: 0.5)')
    parser.add_argument('--save', action='store_true',
                        help='Save output video (only for video source)')
    parser.add_argument('--output', type=str, default='output_detection.mp4',
                        help='Output video path (default: output_detection.mp4)')
    
    args = parser.parse_args()
    
    # Initialize detector
    try:
        detector = YOLOv8Detector(args.model, conf_threshold=args.conf)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    # Process based on source
    if args.source.lower() == 'camera':
        process_camera(detector, camera_id=args.camera_id)
    else:
        process_video(detector, args.source, save_output=args.save, output_path=args.output)
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()