import cv2
import numpy as np
from src.speed_detector import SpeedDetector
from src.vehicle_tracker import SimpleTracker
from ultralytics import YOLO

class VideoProcessor:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        self.tracker = SimpleTracker()
        self.speed_detector = SpeedDetector("calibration_data/camera_profiles.json")
        
    def process_video(self, video_path, output_path=None, speed_limit=50):
        """
        Main processing function with speed detection.
        Returns: violations list and annotated video path
        """
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 30  # Default if can't detect
        
        # Setup video writer for output
        if output_path:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        all_violations = []
        
        print(f"Processing video: {video_path}")
        print(f"FPS: {fps}, Speed limit: {speed_limit} km/h")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLO detection
            results = self.model(frame, conf=0.5)
            
            # Extract detections
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        detections.append([x1, y1, x2, y2, conf, cls])
            
            # Track vehicles across frames
            tracked_dets = self.tracker.update(detections)
            
            # Update speed detector with tracked vehicles
            self.speed_detector.update_tracks(tracked_dets, frame_count)
            
            # Calculate speeds for all vehicles
            speeds = self.speed_detector.get_all_speeds(fps)
            
            # Detect speed violations
            violations = self.speed_detector.detect_speed_violations(speed_limit)
            all_violations.extend(violations)
            
            # Draw annotations on frame
            annotated_frame = self.draw_annotations(frame, tracked_dets, speeds, speed_limit)
            
            if output_path:
                out.write(annotated_frame)
            
            # Print progress
            if frame_count % 100 == 0:
                print(f"Frame {frame_count}: Tracking {len(speeds)} vehicles")
            
            frame_count += 1
        
        cap.release()
        if output_path:
            out.release()
        
        print(f"Processing complete. Found {len(all_violations)} speed violations")
        return all_violations, output_path
    
    def draw_annotations(self, frame, detections, speeds, speed_limit):
        """Draw bounding boxes and speeds on frame."""
        for det in detections:
            if len(det) >= 7:
                x1, y1, x2, y2, conf, cls, track_id = det
                
                # Get color based on speed
                color = (0, 255, 0)  # Green by default
                speed = speeds.get(track_id, 0)
                
                if speed > speed_limit:
                    color = (0, 0, 255)  # Red for speeding
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw speed label
                label = f"ID:{track_id} {speed:.1f}km/h"
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw speed limit indicator
        cv2.putText(frame, f"Speed Limit: {speed_limit} km/h", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def generate_report(self, violations, video_path):
        """Generate a text report of violations."""
        report_path = video_path.replace('.mp4', '_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=== Speed Violation Report ===\n")
            f.write(f"Video: {video_path}\n")
            f.write(f"Total violations: {len(violations)}\n\n")
            
            for i, viol in enumerate(violations, 1):
                f.write(f"Violation #{i}:\n")
                f.write(f"  Vehicle ID: {viol['track_id']}\n")
                f.write(f"  Speed: {viol['speed']:.1f} km/h\n")
                f.write(f"  Speed Limit: {viol['speed_limit']} km/h\n")
                f.write(f"  Excess: {viol['speed'] - viol['speed_limit']:.1f} km/h\n")
                f.write(f"  Class: {self.model.names[viol['class_id']]}\n\n")
        
        return report_path