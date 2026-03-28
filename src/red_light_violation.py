import cv2
import os
import time
import json
import numpy as np
from datetime import datetime
from ultralytics import YOLO

class RedLightViolationDetector:
    def __init__(self, vehicle_model_path, traffic_light_model_path, 
                 plate_model_path=None, config_file="intersection_config.json"):
        print("=" * 60)
        print("RED LIGHT VIOLATION DETECTOR (WITH LICENSE PLATE)")
        print("=" * 60)
        
        # Load models
        print("\nLoading models...")
        self.vehicle_model = YOLO(vehicle_model_path)
        self.traffic_light_model = YOLO(traffic_light_model_path)
        
        # Load license plate model if available
        self.plate_model = None
        if plate_model_path and os.path.exists(plate_model_path):
            self.plate_model = YOLO(plate_model_path)
            print("License plate model loaded")
        else:
            print("No license plate model found. Will use vehicle rear region for plate extraction.")
        
        # Get class names
        self.vehicle_classes = self.vehicle_model.names
        self.traffic_light_classes = self.traffic_light_model.names
        
        print(f"Vehicle classes: {list(self.vehicle_classes.values())}")
        print(f"Traffic light classes: {list(self.traffic_light_classes.values())}")
        
        # Define which traffic light states mean "STOP" (red)
        self.red_light_states = ['red', 'redyellow', 'unknown']
        
        # Store violation data
        self.violations = []
        self.violation_count = 0
        
        # Track vehicles that have already been recorded for this violation
        self.recorded_vehicles = set()  # Track by (frame_range, vehicle_id)
        
        # Create output directories
        self.output_dir = "violations"
        self.screenshots_dir = os.path.join(self.output_dir, "screenshots")
        self.plates_dir = os.path.join(self.output_dir, "license_plates")
        os.makedirs(self.screenshots_dir, exist_ok=True)
        os.makedirs(self.plates_dir, exist_ok=True)
        
        # Video properties
        self.frame_width = None
        self.frame_height = None
        self.fps = None
        
        # Stop line parameters
        self.stop_line_y = None
        self.stop_line_x_start = None
        self.stop_line_x_end = None
        self.detection_method = None
        
        # Configuration
        self.config_file = config_file
        self.intersection_configs = self.load_intersection_configs()
        
        # Lane detection parameters
        self.canny_low = 50
        self.canny_high = 150
        self.hough_threshold = 50
        self.min_line_length = 100
        self.max_line_gap = 50
    
    def _resize_for_display(self, frame, max_width=1280, max_height=720):
        """Resize frame for display only, keeps original for processing"""
        height, width = frame.shape[:2]
        scale = min(max_width / width, max_height / height)
        
        if scale < 1:
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(frame, (new_width, new_height))
        return frame
    
    def load_intersection_configs(self):
        """Load pre-configured intersection settings from JSON file"""
        configs = {}
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    configs = json.load(f)
                print(f"Loaded {len(configs)} intersection configurations")
            except Exception as e:
                print(f"Warning: Could not load config file: {e}")
        return configs
    
    def save_intersection_config(self, intersection_id):
        """Save stop line configuration for this intersection"""
        if not self.stop_line_y:
            return
        
        config = {
            'stop_line': {
                'y': int(self.stop_line_y),
                'x_start': int(self.stop_line_x_start),
                'x_end': int(self.stop_line_x_end)
            },
            'frame_size': {
                'width': self.frame_width,
                'height': self.frame_height
            },
            'detection_method': self.detection_method,
            'saved_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.intersection_configs[intersection_id] = config
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.intersection_configs, f, indent=2)
            print(f"Saved configuration for intersection '{intersection_id}'")
        except Exception as e:
            print(f"Warning: Could not save config: {e}")
    
    def detect_stop_line_auto(self, frame):
        """Automatically detect stop line using lane detection"""
        height, width = frame.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        # Region of interest (bottom half of frame)
        mask = np.zeros_like(edges)
        roi_vertices = np.array([[
            (0, height * 0.5),
            (width, height * 0.5),
            (width, height),
            (0, height)
        ]], dtype=np.int32)
        cv2.fillPoly(mask, roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # Detect lines
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        if lines is None:
            return self.get_default_stop_line(height, width)
        
        # Find horizontal lines
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 20:
                line_y = (y1 + y2) // 2
                line_x1 = min(x1, x2)
                line_x2 = max(x1, x2)
                line_length = line_x2 - line_x1
                
                if line_length > width * 0.3:
                    horizontal_lines.append({
                        'y': line_y,
                        'x1': line_x1,
                        'x2': line_x2,
                        'length': line_length
                    })
        
        if horizontal_lines:
            horizontal_lines.sort(key=lambda l: l['y'], reverse=True)
            for line in horizontal_lines[:3]:
                if line['y'] > height * 0.6:
                    self.stop_line_y = line['y']
                    self.stop_line_x_start = line['x1']
                    self.stop_line_x_end = line['x2']
                    self.detection_method = "lane_detection"
                    print(f"Auto-detected stop line at y={self.stop_line_y}")
                    return True
        
        return self.get_default_stop_line(height, width)
    
    def get_default_stop_line(self, height, width):
        """Set default stop line at 70% of frame height"""
        self.stop_line_y = int(height * 0.7)
        self.stop_line_x_start = 0
        self.stop_line_x_end = width
        self.detection_method = "default"
        print(f"Using default stop line at y={self.stop_line_y}")
        return True
    
    def set_stop_line(self, frame, interactive=False, intersection_id=None):
        """Set stop line automatically or interactively"""
        height, width = frame.shape[:2]
        self.frame_width = width
        self.frame_height = height
        
        # Try config first
        if intersection_id and intersection_id in self.intersection_configs:
            config = self.intersection_configs[intersection_id]
            if 'stop_line' in config:
                self.stop_line_y = config['stop_line']['y']
                self.stop_line_x_start = config['stop_line']['x_start']
                self.stop_line_x_end = config['stop_line']['x_end']
                self.detection_method = "config"
                print(f"Loaded stop line from config: y={self.stop_line_y}")
                return True
        
        # Auto-detect
        if not interactive:
            return self.detect_stop_line_auto(frame)
        
        # Interactive mode
        print("\nSetting stop line interactively...")
        print("Click and drag to draw the stop line, then press SPACE")
        print("Press 'a' for auto-detect")
        
        img_copy = frame.copy()
        drawing = False
        start_point = None
        end_point = None
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal drawing, start_point, end_point, img_copy
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                start_point = (x, y)
                img_copy = frame.copy()
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                img_copy = frame.copy()
                cv2.line(img_copy, start_point, (x, y), (0, 255, 0), 2)
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                end_point = (x, y)
                cv2.line(img_copy, start_point, end_point, (0, 255, 0), 2)
        
        cv2.namedWindow("Set Stop Line", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Set Stop Line", 1280, 720)
        cv2.setMouseCallback("Set Stop Line", mouse_callback)
        
        while True:
            cv2.imshow("Set Stop Line", img_copy)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space
                if start_point and end_point:
                    self.stop_line_y = start_point[1]
                    self.stop_line_x_start = min(start_point[0], end_point[0])
                    self.stop_line_x_end = max(start_point[0], end_point[0])
                    self.detection_method = "manual"
                    if intersection_id:
                        self.save_intersection_config(intersection_id)
                    break
                else:
                    print("Please draw a line first")
            elif key == ord('a'):
                cv2.destroyWindow("Set Stop Line")
                return self.detect_stop_line_auto(frame)
        
        cv2.destroyWindow("Set Stop Line")
        print(f"Stop line set at y={self.stop_line_y}")
        return True
    
    def is_light_red(self, light_state):
        """Check if traffic light state means stop"""
        return light_state in self.red_light_states
    
    def detect_traffic_light(self, frame):
        """Detect traffic light state in frame"""
        results = self.traffic_light_model(frame, conf=0.3, device=0, verbose=False)
        
        light_state = None
        light_confidence = 0
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if conf > light_confidence:
                        light_confidence = conf
                        light_state = self.traffic_light_classes[cls_id]
        
        return light_state, light_confidence
    
    def detect_vehicles(self, frame):
        """Detect vehicles in frame"""
        results = self.vehicle_model(frame, conf=0.4, device=0, verbose=False)
        
        vehicles = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    vehicle_type = self.vehicle_classes[cls_id]
                    
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    vehicles.append({
                        'type': vehicle_type,
                        'confidence': conf,
                        'box': (int(x1), int(y1), int(x2), int(y2)),
                        'bottom_y': int(y2),
                        'center_x': int((x1 + x2) / 2),
                        'center_y': int((y1 + y2) / 2),
                        'width': int(x2 - x1),
                        'height': int(y2 - y1)
                    })
        
        return vehicles
    
    def extract_license_plate(self, frame, vehicle):
        """Extract license plate region from vehicle"""
        x1, y1, x2, y2 = vehicle['box']
        
        # Focus on bottom portion of vehicle where plates typically are
        plate_y1 = int(y2 - vehicle['height'] * 0.3)  # Bottom 30%
        plate_y2 = y2
        plate_x1 = x1
        plate_x2 = x2
        
        # Add some margin
        margin_x = int(vehicle['width'] * 0.1)
        margin_y = int(vehicle['height'] * 0.05)
        
        plate_x1 = max(0, x1 - margin_x)
        plate_x2 = min(self.frame_width, x2 + margin_x)
        plate_y1 = max(0, plate_y1 - margin_y)
        plate_y2 = min(self.frame_height, y2 + margin_y)
        
        # Extract plate region
        plate_region = frame[plate_y1:plate_y2, plate_x1:plate_x2].copy()
        
        # If plate model exists, run detection
        if self.plate_model:
            plate_results = self.plate_model(plate_region, conf=0.3, verbose=False)
            # Process plate results here if needed
        
        return plate_region, (plate_x1, plate_y1, plate_x2, plate_y2)
    
    def check_violation(self, vehicle, light_state, frame_count):
        """Check if vehicle crossed stop line when light was red"""
        if not self.is_light_red(light_state):
            return False
        
        # Check if vehicle crossed the line
        if vehicle['bottom_y'] > self.stop_line_y:
            if self.stop_line_x_start <= vehicle['center_x'] <= self.stop_line_x_end:
                # Create a unique ID for this violation event
                # Use vehicle position rounded to 50px to group nearby detections
                grid_x = round(vehicle['center_x'] / 50) * 50
                grid_y = round(vehicle['center_y'] / 50) * 50
                vehicle_key = f"{grid_x}_{grid_y}"
                
                # Only record if not already recorded in last 30 frames
                if vehicle_key not in self.recorded_vehicles:
                    self.recorded_vehicles.add(vehicle_key)
                    return True
        
        return False
    
    def save_violation(self, frame, vehicle, light_state, light_confidence, frame_count):
        """Save ONE screenshot per violation with vehicle and plate"""
        self.violation_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Save full vehicle screenshot
        vehicle_filename = f"violation_{self.violation_count:03d}_vehicle_{timestamp}.jpg"
        vehicle_path = os.path.join(self.screenshots_dir, vehicle_filename)
        
        # Annotate full frame
        annotated = frame.copy()
        
        # Draw stop line
        cv2.line(annotated, 
                 (self.stop_line_x_start, self.stop_line_y),
                 (self.stop_line_x_end, self.stop_line_y),
                 (0, 255, 0), 3)
        
        # Draw vehicle
        x1, y1, x2, y2 = vehicle['box']
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(annotated, f"{vehicle['type']} ({vehicle['confidence']:.2f})",
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Add violation info
        cv2.putText(annotated, f"VIOLATION #{self.violation_count}", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(annotated, f"Light: {light_state} ({light_confidence:.2f})",
                   (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated, f"Frame: {frame_count}",
                   (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imwrite(vehicle_path, annotated)
        
        # 2. Extract and save license plate region
        plate_region, plate_box = self.extract_license_plate(frame, vehicle)
        plate_filename = f"violation_{self.violation_count:03d}_plate_{timestamp}.jpg"
        plate_path = os.path.join(self.plates_dir, plate_filename)
        cv2.imwrite(plate_path, plate_region)
        
        # 3. Save violation data
        violation_data = {
            'id': self.violation_count,
            'timestamp': timestamp,
            'frame': frame_count,
            'vehicle_type': vehicle['type'],
            'vehicle_confidence': vehicle['confidence'],
            'light_state': light_state,
            'light_confidence': light_confidence,
            'vehicle_screenshot': vehicle_filename,
            'plate_screenshot': plate_filename,
            'stop_line_y': self.stop_line_y
        }
        self.violations.append(violation_data)
        
        print(f"\n  VIOLATION #{self.violation_count} DETECTED")
        print(f"    Vehicle: {vehicle['type']}")
        print(f"    Vehicle image: {vehicle_path}")
        print(f"    License plate image: {plate_path}")
        
        return vehicle_path, plate_path
    
    def process_video(self, video_path, output_video_path=None, intersection_id=None, interactive=False):
        """Process video for red light violations"""
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
        
        # Read first frame for stop line setup
        ret, first_frame = cap.read()
        if not ret:
            print("Error: Could not read first frame")
            return
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Set stop line
        if not self.set_stop_line(first_frame, interactive, intersection_id):
            print("Failed to set stop line")
            return
        
        # Reset recorded vehicles for this video
        self.recorded_vehicles = set()
        
        # Setup output video
        if output_video_path is None:
            video_name = os.path.basename(video_path)
            name_without_ext = os.path.splitext(video_name)[0]
            output_video_path = os.path.join(self.output_dir, f"{name_without_ext}_analyzed.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, self.fps, 
                             (self.frame_width, self.frame_height))
        
        # Process video
        frame_count = 0
        current_light_state = None
        current_light_confidence = 0
        
        print("\nProcessing video for violations...")
        print("Press 'q' to quit, 'p' to pause")
        print(f"Stop line detection method: {self.detection_method}")
        
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Progress every 30 frames
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {frame_count}/{total_frames} ({progress:.1f}%) - Violations: {self.violation_count}")
                
                # Detect traffic light
                light_state, light_confidence = self.detect_traffic_light(frame)
                if light_state:
                    current_light_state = light_state
                    current_light_confidence = light_confidence
                
                # Detect vehicles
                vehicles = self.detect_vehicles(frame)
                
                # Annotate frame
                annotated = frame.copy()
                
                # Draw stop line
                cv2.line(annotated, 
                         (self.stop_line_x_start, self.stop_line_y),
                         (self.stop_line_x_end, self.stop_line_y),
                         (0, 255, 0), 2)
                
                # Draw traffic light state
                if current_light_state:
                    color = (0, 0, 255) if self.is_light_red(current_light_state) else (0, 255, 0)
                    cv2.putText(annotated, 
                               f"LIGHT: {current_light_state.upper()} ({current_light_confidence:.2f})",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Check each vehicle
                for vehicle in vehicles:
                    x1, y1, x2, y2 = vehicle['box']
                    
                    # Check for violation
                    is_violation = self.check_violation(vehicle, current_light_state, frame_count)
                    
                    # Draw vehicle
                    color = (0, 0, 255) if is_violation else (0, 255, 0)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated, 
                               f"{vehicle['type']} ({vehicle['confidence']:.2f})",
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Save violation (only once per vehicle)
                    if is_violation:
                        self.save_violation(frame, vehicle, current_light_state, 
                                          current_light_confidence, frame_count)
                        
                        # Mark on frame
                        cv2.putText(annotated, f"VIOLATION #{self.violation_count}", 
                                   (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Write to output video
                out.write(annotated)
                
                # Show frame
                display_frame = self._resize_for_display(annotated)
                cv2.imshow("Red Light Violation Detection", display_frame)
            
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
        print(f"Total violations detected: {self.violation_count}")
        print(f"Vehicle screenshots: {self.screenshots_dir}")
        print(f"License plate images: {self.plates_dir}")
        print(f"Analyzed video: {output_video_path}")
        print(f"Report: {os.path.join(self.output_dir, 'violation_report.json')}")
    
    def save_report(self, video_path):
        """Save violation report with proper JSON serialization"""
        # Convert all values to JSON serializable types
        serializable_violations = []
        for v in self.violations:
            serializable_v = {}
            for key, value in v.items():
                # Convert numpy types to Python native types
                if hasattr(value, 'item'):
                    serializable_v[key] = value.item()
                elif isinstance(value, (np.integer, np.floating, np.ndarray)):
                    serializable_v[key] = value.tolist() if hasattr(value, 'tolist') else float(value)
                else:
                    serializable_v[key] = value
            serializable_violations.append(serializable_v)
        
        report = {
            'video_file': str(video_path),
            'processing_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_violations': int(self.violation_count),
            'detection_method': str(self.detection_method) if self.detection_method else None,
            'stop_line': {
                'y': int(self.stop_line_y) if self.stop_line_y else None,
                'x_start': int(self.stop_line_x_start) if self.stop_line_x_start else None,
                'x_end': int(self.stop_line_x_end) if self.stop_line_x_end else None
            },
            'violations': serializable_violations
        }
        
        report_path = os.path.join(self.output_dir, 'violation_report.json')
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\nReport saved: {report_path}")
        except Exception as e:
            print(f"Warning: Could not save report: {e}")
            # Save a simplified version without violations if error persists
            simple_report = {
                'video_file': str(video_path),
                'processing_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'total_violations': int(self.violation_count)
            }
            simple_path = report_path.replace('.json', '_simple.json')
            with open(simple_path, 'w') as f:
                json.dump(simple_report, f, indent=2)
            print(f"Simplified report saved: {simple_path}")

def main():
    # Model paths
    vehicle_model_path = "models/yolov8n_trained/weights/best.pt"
    traffic_light_model_path = "models/traffic_light_detector/weights/best.pt"
    plate_model_path = "models/license_plate_model.pt"  # Optional
    
    # Check models
    if not os.path.exists(vehicle_model_path):
        print(f"Error: Vehicle model not found at {vehicle_model_path}")
        return
    
    if not os.path.exists(traffic_light_model_path):
        print(f"Error: Traffic light model not found at {traffic_light_model_path}")
        return
    
    print("\nRED LIGHT VIOLATION DETECTION")
    print("=" * 60)
    print("1. Automatic mode (detects stop line automatically)")
    print("2. Interactive mode (draw stop line manually)")
    print("3. Use pre-configured intersection")
    
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
    detector = RedLightViolationDetector(vehicle_model_path, traffic_light_model_path, plate_model_path)
    
    # Process based on mode
    if mode == '1':
        print("\nAutomatic mode: Stop line will be detected automatically")
        detector.process_video(video_path, interactive=False)
    elif mode == '2':
        print("\nInteractive mode: You will draw the stop line")
        detector.process_video(video_path, interactive=True)
    elif mode == '3':
        intersection_id = input("Enter intersection ID: ").strip()
        detector.process_video(video_path, intersection_id=intersection_id, interactive=False)
    else:
        print("Invalid mode. Using automatic.")
        detector.process_video(video_path, interactive=False)

if __name__ == "__main__":
    main()