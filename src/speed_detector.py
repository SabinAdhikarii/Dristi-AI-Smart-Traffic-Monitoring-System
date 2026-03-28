import cv2
import os
import json
import numpy as np
import time
from datetime import datetime
from ultralytics import YOLO
from collections import defaultdict

class SpeedDetector:
    def __init__(self, vehicle_model_path, config_file="speed_config.json"):
        print("=" * 60)
        print("SPEED DETECTION SYSTEM")
        print("=" * 60)
        
        # Load vehicle detection model
        print("\nLoading vehicle model...")
        self.vehicle_model = YOLO(vehicle_model_path)
        self.vehicle_classes = self.vehicle_model.names
        print(f"Vehicle classes: {list(self.vehicle_classes.values())}")
        
        # Speed tracking variables
        self.vehicle_tracks = defaultdict(lambda: {
            'positions': [],      # List of (frame, x, y) positions
            'speeds': [],         # Calculated speeds
            'last_seen': 0,
            'violation_recorded': False
        })
        
        # Calibration parameters
        self.calibration = self.load_calibration(config_file)
        self.pixels_per_meter = self.calibration.get('pixels_per_meter', 10.0)
        self.speed_limit = self.calibration.get('speed_limit', 60)  # km/h
        self.distance_markers = self.calibration.get('distance_markers', [])
        
        # Detection zone (where speed is calculated)
        self.zone_start = None
        self.zone_end = None
        self.zone_y = None
        
        # Output directories
        self.output_dir = "violations"
        self.screenshots_dir = os.path.join(self.output_dir, "speed_violations")
        os.makedirs(self.screenshots_dir, exist_ok=True)
        
        # Statistics
        self.violations = []
        self.violation_count = 0
        self.total_vehicles_tracked = 0
        
        # Video properties
        self.fps = None
        self.frame_width = None
        self.frame_height = None
        
        # Tracking parameters
        self.tracking_distance = 50  # pixels
        self.max_track_frames = 30   # frames to keep track
        
    def load_calibration(self, config_file):
        """Load calibration parameters from JSON file"""
        default_config = {
            'pixels_per_meter': 10.0,
            'speed_limit': 60,
            'distance_markers': [],
            'calibration_date': None
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                print(f"Loaded calibration from {config_file}")
                return config
            except Exception as e:
                print(f"Warning: Could not load calibration: {e}")
        
        return default_config
    
    def save_calibration(self, config_file="speed_config.json"):
        """Save calibration parameters"""
        self.calibration['pixels_per_meter'] = self.pixels_per_meter
        self.calibration['speed_limit'] = self.speed_limit
        self.calibration['calibration_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            with open(config_file, 'w') as f:
                json.dump(self.calibration, f, indent=2)
            print(f"Calibration saved to {config_file}")
        except Exception as e:
            print(f"Warning: Could not save calibration: {e}")
    
    def calibrate_interactive(self, frame):
        """
        Interactive calibration - user selects two points of known distance
        """
        print("\n" + "=" * 60)
        print("INTERACTIVE CALIBRATION")
        print("=" * 60)
        print("Click two points on the road that are 10 meters apart")
        print("Press SPACE after selecting both points")
        print("Press 'c' to cancel")
        
        img_copy = frame.copy()
        points = []
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal img_copy, points
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
                points.append((x, y))
                cv2.circle(img_copy, (x, y), 5, (0, 255, 0), -1)
                if len(points) == 2:
                    cv2.line(img_copy, points[0], points[1], (0, 255, 0), 2)
                    # Calculate distance in pixels
                    dist_pixels = np.sqrt((points[1][0] - points[0][0])**2 + 
                                         (points[1][1] - points[0][1])**2)
                    cv2.putText(img_copy, f"Distance: {dist_pixels:.1f} pixels", 
                               (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Calibration", 1280, 720)
        cv2.setMouseCallback("Calibration", mouse_callback)
        
        while True:
            cv2.imshow("Calibration", img_copy)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' ') and len(points) == 2:
                # Calculate pixels per meter (assuming 10 meters distance)
                dist_pixels = np.sqrt((points[1][0] - points[0][0])**2 + 
                                     (points[1][1] - points[0][1])**2)
                self.pixels_per_meter = dist_pixels / 10.0  # 10 meters reference
                print(f"\nCalibration complete:")
                print(f"  Pixel distance: {dist_pixels:.1f} pixels")
                print(f"  Pixels per meter: {self.pixels_per_meter:.2f}")
                break
            elif key == ord('c'):
                print("Calibration cancelled")
                cv2.destroyWindow("Calibration")
                return False
        
        cv2.destroyWindow("Calibration")
        
        # Ask for speed limit
        try:
            speed_input = input(f"\nEnter speed limit (km/h) [default: {self.speed_limit}]: ").strip()
            if speed_input:
                self.speed_limit = float(speed_input)
        except:
            pass
        
        self.save_calibration()
        return True
    
    def set_detection_zone(self, frame):
        """
        Set the zone where speed is calculated
        """
        height, width = frame.shape[:2]
        
        # Default zone: horizontal line in bottom half
        print("\nSetting speed detection zone...")
        print("Default zone: horizontal line across frame")
        
        # Let user choose between line and region
        print("\nSelect detection method:")
        print("1. Single line (simpler, good for perpendicular traffic)")
        print("2. Region between two lines (more accurate)")
        
        choice = input("Enter choice (1/2) [default: 1]: ").strip() or "1"
        
        if choice == "1":
            # Single line method
            self.zone_y = int(height * 0.7)
            self.zone_start = 0
            self.zone_end = width
            print(f"Detection line set at y = {self.zone_y}")
        else:
            # Two-line region
            print("\nClick two points to define the detection region start line")
            print("Then click two points to define the end line")
            self._set_region_interactive(frame)
        
        return True
    
    def _set_region_interactive(self, frame):
        """Interactive region selection"""
        img_copy = frame.copy()
        lines = []
        current_line = []
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal img_copy, lines, current_line
            if event == cv2.EVENT_LBUTTONDOWN:
                current_line.append((x, y))
                cv2.circle(img_copy, (x, y), 5, (0, 255, 0), -1)
                
                if len(current_line) == 2:
                    lines.append(current_line.copy())
                    cv2.line(img_copy, current_line[0], current_line[1], (0, 255, 0), 2)
                    current_line = []
        
        cv2.namedWindow("Set Detection Region", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Set Detection Region", 1280, 720)
        cv2.setMouseCallback("Set Detection Region", mouse_callback)
        
        print("Click two points for start line, then two points for end line")
        
        while len(lines) < 2:
            cv2.imshow("Set Detection Region", img_copy)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyWindow("Set Detection Region")
        
        if len(lines) == 2:
            # Store region boundaries
            self.zone_start = lines[0]
            self.zone_end = lines[1]
            print("Detection region set")
    
    def pixels_to_meters(self, pixels):
        """Convert pixel distance to meters"""
        return pixels / self.pixels_per_meter
    
    def calculate_speed(self, track_id, current_pos, frame_count):
        """
        Calculate speed based on position change over time
        Returns speed in km/h
        """
        track = self.vehicle_tracks[track_id]
        
        # Add current position
        track['positions'].append((frame_count, current_pos[0], current_pos[1]))
        track['last_seen'] = frame_count
        
        # Keep only recent positions
        if len(track['positions']) > self.max_track_frames:
            track['positions'].pop(0)
        
        # Need at least 2 positions to calculate speed
        if len(track['positions']) < 2:
            return 0
        
        # Get first and last position in track
        first_frame, first_x, first_y = track['positions'][0]
        last_frame, last_x, last_y = track['positions'][-1]
        
        # Calculate frame difference
        frame_diff = last_frame - first_frame
        if frame_diff == 0:
            return 0
        
        # Calculate pixel distance moved
        pixel_distance = np.sqrt((last_x - first_x)**2 + (last_y - first_y)**2)
        
        # Convert to meters
        meters = self.pixels_to_meters(pixel_distance)
        
        # Calculate time in seconds
        time_seconds = frame_diff / self.fps
        
        # Calculate speed in m/s then convert to km/h
        speed_ms = meters / time_seconds if time_seconds > 0 else 0
        speed_kmh = speed_ms * 3.6
        
        # Store speed
        track['speeds'].append(speed_kmh)
        
        return speed_kmh
    
    def is_overspeeding(self, speed):
        """Check if vehicle is overspeeding"""
        return speed > self.speed_limit
    
    def save_violation(self, frame, vehicle, track_id, speed, light_state=None):
        """Save speeding violation screenshot"""
        self.violation_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare filename
        filename = f"speeding_{self.violation_count:03d}_{vehicle['type']}_{timestamp}.jpg"
        filepath = os.path.join(self.screenshots_dir, filename)
        
        # Annotate frame
        annotated = frame.copy()
        
        # Draw vehicle
        x1, y1, x2, y2 = vehicle['box']
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        # Add speed information
        cv2.putText(annotated, f"SPEEDING VIOLATION", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(annotated, f"Vehicle: {vehicle['type']}", 
                   (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated, f"Speed: {speed:.1f} km/h (Limit: {self.speed_limit})", 
                   (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(annotated, f"Track ID: {track_id}", 
                   (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Save screenshot
        cv2.imwrite(filepath, annotated)
        
        # Record violation
        violation_data = {
            'id': self.violation_count,
            'timestamp': timestamp,
            'track_id': track_id,
            'vehicle_type': vehicle['type'],
            'speed': round(speed, 1),
            'speed_limit': self.speed_limit,
            'screenshot': filename
        }
        self.violations.append(violation_data)
        
        print(f"\n  SPEEDING VIOLATION #{self.violation_count}")
        print(f"    Vehicle: {vehicle['type']} (Track {track_id})")
        print(f"    Speed: {speed:.1f} km/h (Limit: {self.speed_limit})")
        print(f"    Screenshot: {filepath}")
        
        return filepath
    
    def process_video(self, video_path, output_video_path=None, calibrate=False):
        """Process video for speed detection"""
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
        
        # Read first frame for setup
        ret, first_frame = cap.read()
        if not ret:
            print("Error: Could not read first frame")
            return
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Calibration if requested
        if calibrate:
            if not self.calibrate_interactive(first_frame):
                return
        
        # Set detection zone
        self.set_detection_zone(first_frame)
        
        # Setup output video
        if output_video_path is None:
            video_name = os.path.basename(video_path)
            name_without_ext = os.path.splitext(video_name)[0]
            output_video_path = os.path.join(self.output_dir, f"{name_without_ext}_speed_analyzed.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, self.fps, 
                             (self.frame_width, self.frame_height))
        
        # Process video
        frame_count = 0
        track_id_counter = 0
        
        print("\nProcessing video for speed detection...")
        print("Press 'q' to quit, 'p' to pause")
        
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Progress update
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {frame_count}/{total_frames} ({progress:.1f}%) - Vehicles tracked: {len(self.vehicle_tracks)}")
                
                # Detect and track vehicles
                results = self.vehicle_model.track(frame, persist=True, verbose=False, device=0)
                
                # Annotate frame
                annotated = frame.copy()
                
                # Draw detection zone
                if self.zone_y is not None:
                    # Single line mode
                    cv2.line(annotated, (self.zone_start, self.zone_y), 
                            (self.zone_end, self.zone_y), (0, 255, 0), 2)
                    cv2.putText(annotated, "SPEED DETECTION ZONE", 
                               (self.zone_start, self.zone_y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Process detections
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy()
                    classes = results[0].boxes.cls.cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy()
                    
                    for i, (box, track_id, cls, conf) in enumerate(zip(boxes, track_ids, classes, confs)):
                        track_id = int(track_id)
                        x1, y1, x2, y2 = map(int, box)
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        vehicle_type = self.vehicle_classes[int(cls)]
                        
                        # Calculate speed
                        speed = self.calculate_speed(track_id, (center_x, center_y), frame_count)
                        
                        # Check if vehicle is in detection zone
                        in_zone = False
                        if self.zone_y is not None:
                            # Single line: check if vehicle crossed or is near line
                            if abs(center_y - self.zone_y) < 20:
                                in_zone = True
                        
                        # Check for overspeeding
                        is_speeding = self.is_overspeeding(speed) and speed > 0
                        
                        # Draw vehicle
                        color = (0, 0, 255) if is_speeding else (0, 255, 0)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        
                        # Display information
                        info_text = f"ID:{track_id} {vehicle_type} {speed:.1f}km/h"
                        if is_speeding:
                            info_text += " SPEEDING!"
                        
                        cv2.putText(annotated, info_text, (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Save violation if speeding and not already recorded
                        if is_speeding and not self.vehicle_tracks[track_id]['violation_recorded']:
                            vehicle_info = {
                                'type': vehicle_type,
                                'confidence': conf,
                                'box': (x1, y1, x2, y2)
                            }
                            self.save_violation(frame, vehicle_info, track_id, speed)
                            self.vehicle_tracks[track_id]['violation_recorded'] = True
                            self.total_vehicles_tracked += 1
                
                # Add speed limit info
                cv2.putText(annotated, f"Speed Limit: {self.speed_limit} km/h", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated, f"Violations: {self.violation_count}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Write frame
                out.write(annotated)
                
                # Show frame
                display_frame = cv2.resize(annotated, (1280, 720))
                cv2.imshow("Speed Detection", display_frame)
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Processing stopped by user")
                break
            elif key == ord('p'):
                paused = not paused
        
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Save report
        self.save_report(video_path)
        
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Total vehicles tracked: {self.total_vehicles_tracked}")
        print(f"Speeding violations: {self.violation_count}")
        print(f"Output video: {output_video_path}")
        print(f"Violation screenshots: {self.screenshots_dir}")
    
    def save_report(self, video_path):
        """Save speed detection report"""
        report = {
            'video_file': video_path,
            'processing_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'fps': self.fps,
            'calibration': {
                'pixels_per_meter': self.pixels_per_meter,
                'speed_limit': self.speed_limit
            },
            'total_vehicles_tracked': self.total_vehicles_tracked,
            'speeding_violations': self.violation_count,
            'violations': self.violations
        }
        
        report_path = os.path.join(self.output_dir, 'speed_report.json')
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\nReport saved: {report_path}")
        except Exception as e:
            print(f"Warning: Could not save report: {e}")

def main():
    # Model path
    vehicle_model_path = "models/yolov8n_trained/weights/best.pt"
    
    # Check model
    if not os.path.exists(vehicle_model_path):
        print(f"Error: Vehicle model not found at {vehicle_model_path}")
        return
    
    print("\nSPEED DETECTION SYSTEM")
    print("=" * 60)
    print("1. Quick mode (use existing calibration)")
    print("2. Calibrate and process")
    print("3. View calibration settings")
    
    mode = input("\nSelect mode (1/2/3): ").strip()
    
    # Get video file
    print("\nAvailable test videos:")
    test_videos_dir = "data/test_videos"
    videos = []
    
    if os.path.exists(test_videos_dir):
        videos = [f for f in os.listdir(test_videos_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        for i, video in enumerate(videos, 1):
            print(f"  {i}. {video}")
    
    choice = input("\nSelect video number or enter path: ").strip()
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(videos):
            video_path = os.path.join(test_videos_dir, videos[idx])
        else:
            video_path = choice
    except ValueError:
        video_path = choice
    
    if not os.path.exists(video_path):
        print(f"Error: Video not found: {video_path}")
        return
    
    # Create detector
    detector = SpeedDetector(vehicle_model_path)
    
    # Process based on mode
    if mode == '1':
        print("\nQuick mode: Using existing calibration")
        detector.process_video(video_path, calibrate=False)
    elif mode == '2':
        print("\nCalibration mode: Please follow instructions")
        detector.process_video(video_path, calibrate=True)
    elif mode == '3':
        print(f"\nCurrent calibration:")
        print(f"  Pixels per meter: {detector.pixels_per_meter}")
        print(f"  Speed limit: {detector.speed_limit} km/h")
    else:
        print("Invalid mode. Using quick mode.")
        detector.process_video(video_path, calibrate=False)

if __name__ == "__main__":
    main()