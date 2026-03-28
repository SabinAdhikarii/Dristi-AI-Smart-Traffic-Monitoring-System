import cv2
import os
import json
import numpy as np
from datetime import datetime
from ultralytics import YOLO

class HelmetDetector:
    def __init__(self, model_path):
        print("=" * 60)
        print("HELMET DETECTION SYSTEM")
        print("=" * 60)
        
        # Load model
        print(f"\nLoading model from: {model_path}")
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        print(f"Classes: {list(self.class_names.values())}")
        
        # Define violation classes (riders without helmet)
        # Based on your dataset classes
        self.helmet_classes = ['driver_with_helmet', 'passenger_with_helemt']
        self.no_helmet_classes = ['driver_without_helmet', 'passenger_without_helemt']
        self.motorcycle_classes = ['bike']
        self.rider_classes = ['driver', 'passenger']
        
        # Output directories
        self.output_dir = "violations"
        self.helmet_dir = os.path.join(self.output_dir, "helmet_violations")
        os.makedirs(self.helmet_dir, exist_ok=True)
        
        # Statistics
        self.violations = []
        self.violation_count = 0
        self.motorcycles_detected = 0
        self.riders_checked = 0
        
        # Video properties
        self.fps = None
        self.frame_width = None
        self.frame_height = None
        
        # Track recorded violations to avoid duplicates
        self.recorded_violations = set()
    
    def _resize_for_display(self, frame, max_width=1280, max_height=720):
        """Resize frame for display"""
        height, width = frame.shape[:2]
        scale = min(max_width / width, max_height / height)
        
        if scale < 1:
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(frame, (new_width, new_height))
        return frame
    
    def is_violation(self, class_name):
        """Check if class indicates no helmet violation"""
        return class_name in self.no_helmet_classes
    
    def is_motorcycle(self, class_name):
        """Check if class is motorcycle"""
        return class_name in self.motorcycle_classes
    
    def is_rider(self, class_name):
        """Check if class is a rider"""
        return class_name in self.rider_classes or class_name in self.helmet_classes or class_name in self.no_helmet_classes
    
    def save_violation(self, frame, box, class_name, confidence, frame_count):
        """Save helmet violation screenshot"""
        self.violation_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        x1, y1, x2, y2 = box
        
        # Create unique ID for this violation
        violation_id = f"violation_{self.violation_count:03d}"
        
        # Save full frame with annotation
        filename = f"{violation_id}_{timestamp}.jpg"
        filepath = os.path.join(self.helmet_dir, filename)
        
        # Annotate frame
        annotated = frame.copy()
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        # Add labels
        label = f"NO HELMET - {class_name}"
        cv2.putText(annotated, label, (x1, y1 - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(annotated, f"Conf: {confidence:.2f}", (x1, y1 - 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Add timestamp
        cv2.putText(annotated, f"Violation #{self.violation_count}", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(annotated, f"Frame: {frame_count}", 
                   (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Save image
        cv2.imwrite(filepath, annotated)
        
        # Also save cropped rider for closer inspection
        rider_filename = f"{violation_id}_rider_{timestamp}.jpg"
        rider_path = os.path.join(self.helmet_dir, rider_filename)
        rider_crop = frame[y1:y2, x1:x2].copy()
        cv2.imwrite(rider_path, rider_crop)
        
        # Record violation data
        violation_data = {
            'id': self.violation_count,
            'timestamp': timestamp,
            'frame': frame_count,
            'class': class_name,
            'confidence': float(confidence),
            'screenshot': filename,
            'rider_crop': rider_filename,
            'box': [int(x1), int(y1), int(x2), int(y2)]
        }
        self.violations.append(violation_data)
        
        print(f"\n  VIOLATION #{self.violation_count}: {class_name}")
        print(f"    Confidence: {confidence:.2f}")
        print(f"    Screenshot: {filename}")
        
        return filepath
    
    def process_video(self, video_path, output_video_path=None):
        """Process video for helmet violations"""
        print(f"\nProcessing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {self.frame_width}x{self.frame_height}, {self.fps} FPS, {total_frames} frames")
        
        # Setup output video
        if output_video_path is None:
            video_name = os.path.basename(video_path)
            name_without_ext = os.path.splitext(video_name)[0]
            output_video_path = os.path.join(self.output_dir, f"{name_without_ext}_helmet_analyzed.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, self.fps, 
                             (self.frame_width, self.frame_height))
        
        # Process video
        frame_count = 0
        self.recorded_violations = set()
        
        print("\nProcessing video for helmet violations...")
        print("Press 'q' to quit, 'p' to pause")
        
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Progress update every 30 frames
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {frame_count}/{total_frames} ({progress:.1f}%) - Violations: {self.violation_count}")
                
                # Run detection
                results = self.model(frame, conf=0.25, device=0, verbose=False)
                
                # Annotate frame
                annotated = frame.copy()
                
                # Counters for this frame
                motorcycles = 0
                riders = 0
                
                # Process detections
                if results[0].boxes is not None:
                    boxes = results[0].boxes
                    
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = self.class_names[cls_id]
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        
                        # Count motorcycle
                        if self.is_motorcycle(class_name):
                            motorcycles += 1
                            self.motorcycles_detected += 1
                            color = (255, 255, 0)  # Cyan for bikes
                        
                        # Count and check riders
                        elif self.is_rider(class_name):
                            riders += 1
                            self.riders_checked += 1
                            
                            # Check for violation (no helmet)
                            if self.is_violation(class_name):
                                color = (0, 0, 255)  # Red for violation
                                
                                # Create unique key for this rider (based on position and frame range)
                                rider_key = f"{class_name}_{x1//50}_{y1//50}_{frame_count//30}"
                                
                                # Save violation only once per rider
                                if rider_key not in self.recorded_violations:
                                    self.save_violation(frame, (x1, y1, x2, y2), class_name, conf, frame_count)
                                    self.recorded_violations.add(rider_key)
                            else:
                                color = (0, 255, 0)  # Green for helmet worn
                        
                        else:
                            color = (255, 255, 255)  # White for others
                        
                        # Draw bounding box
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        
                        # Add label
                        label = f"{class_name} {conf:.2f}"
                        cv2.putText(annotated, label, (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Add statistics to frame
                cv2.putText(annotated, f"Helmet Violations: {self.violation_count}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(annotated, f"Motorcycles: {motorcycles}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(annotated, f"Riders Checked: {riders}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Write to output video
                out.write(annotated)
                
                # Show frame (resized)
                display_frame = self._resize_for_display(annotated)
                cv2.imshow("Helmet Detection", display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Processing stopped by user")
                break
            elif key == ord('p'):
                paused = not paused
                print("Paused" if paused else "Resumed")
        
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Save report
        self.save_report(video_path, output_video_path)
        
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Total motorcycles detected: {self.motorcycles_detected}")
        print(f"Total riders checked: {self.riders_checked}")
        print(f"Helmet violations: {self.violation_count}")
        print(f"Output video: {output_video_path}")
        print(f"Violation screenshots: {self.helmet_dir}")
    
    def save_report(self, video_path, output_video_path):
        """Save violation report"""
        report = {
            'video_file': video_path,
            'output_video': output_video_path,
            'processing_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'video_properties': {
                'fps': self.fps,
                'width': self.frame_width,
                'height': self.frame_height
            },
            'statistics': {
                'total_motorcycles': self.motorcycles_detected,
                'total_riders_checked': self.riders_checked,
                'helmet_violations': self.violation_count
            },
            'violations': self.violations
        }
        
        report_path = os.path.join(self.output_dir, 'helmet_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nReport saved: {report_path}")

def main():
    # Model path
    model_path = "models/helmet_detector_kaggle/weights/best.pt"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using train_helmet_kaggle.py")
        return
    
    print("\nHELMET DETECTION FOR MOTORCYCLE RIDERS")
    print("=" * 60)
    
    # Get video file
    print("\nAvailable test videos:")
    test_videos_dir = "data/test_videos"
    
    if not os.path.exists(test_videos_dir):
        os.makedirs(test_videos_dir)
        print(f"Created {test_videos_dir} folder")
        print("Please add video files to test")
        return
    
    videos = [f for f in os.listdir(test_videos_dir) if f.endswith(('.mp4', '.avi', '.mov', '.MP4'))]
    
    if not videos:
        print(f"No videos found in {test_videos_dir}")
        print("Please add video files to test")
        return
    
    # Show available videos
    for i, video in enumerate(videos, 1):
        print(f"  {i}. {video}")
    
    # Get user choice
    choice = input("\nSelect video number (or enter full path): ").strip()
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(videos):
            video_path = os.path.join(test_videos_dir, videos[idx])
        else:
            video_path = choice
    except ValueError:
        video_path = choice
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Error: Video not found: {video_path}")
        return
    
    # Ask for output video name
    default_output = os.path.join("outputs", f"helmet_{os.path.basename(video_path)}")
    output_choice = input(f"Output video path [default: {default_output}]: ").strip()
    output_path = output_choice if output_choice else default_output
    
    # Create detector and process
    detector = HelmetDetector(model_path)
    detector.process_video(video_path, output_path)

if __name__ == "__main__":
    main()