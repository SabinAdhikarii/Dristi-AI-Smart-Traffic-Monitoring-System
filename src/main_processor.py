"""
Main Traffic Monitoring System - GPU Optimized with Database Storage
Includes Red Light, Helmet, Speed Detection, and License Plate OCR
"""
import os
import json
import cv2
import torch
from datetime import datetime
from ultralytics import YOLO
from db import save_violation, get_db_connection
from plate_detector import LicensePlateDetector

# Check GPU
DEVICE = 0 if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU (slower)")

def _dedup_violations(violation_list, frame_window=30, x_tolerance=160):
    if not violation_list:
        return violation_list

    sorted_v = sorted(violation_list, key=lambda v: v.get('frame', 0))
    kept = []

    for candidate in sorted_v:
        c_frame = candidate.get('frame', 0)
        c_track = candidate.get('track_id')
        is_duplicate = False

        for kept_v in kept:
            k_frame = kept_v.get('frame', 0)
            k_track = kept_v.get('track_id')

            if c_track and k_track and c_track == k_track:
                is_duplicate = True
                break

            if abs(c_frame - k_frame) <= frame_window:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(candidate)

    return kept


class TrafficMonitoringSystem:
    def __init__(self):
        print("Initializing Traffic Monitoring System")
        
        # Model paths
        self.vehicle_model_path = "models/yolov8n_trained/weights/best.pt"
        self.traffic_light_model_path = "models/traffic_light_detector/weights/best.pt"
        self.helmet_model_path = "models/helmet_detector_kaggle/weights/best.pt"
        
        # Output directories
        self.uploads_dir = "data/uploaded_videos"
        self.results_base = "results"
        os.makedirs(self.uploads_dir, exist_ok=True)
        os.makedirs(self.results_base, exist_ok=True)
        
        # Models will be loaded when processing starts
        self.vehicle_model = None
        self.traffic_light_model = None
        self.helmet_model = None
        
        # License plate detector
        self.plate_detector = None
        
        # Speed detection settings
        self.speed_limit = 60  # km/h
        self.calibration_pixels_per_meter = 10  # Pixels per meter (adjust via calibration)
        
        # Load calibration from config if exists
        self.load_calibration()
    
    def load_calibration(self):
        """Load speed calibration from config file"""
        config_path = "speed_config.json"
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.speed_limit = config.get('speed_limit', 60)
                    self.calibration_pixels_per_meter = config.get('pixels_per_meter', 10)
                    print(f"Loaded calibration: {self.calibration_pixels_per_meter} px/m, limit: {self.speed_limit} km/h")
            except:
                pass
    
    def save_calibration(self):
        """Save current calibration to config file"""
        config = {
            'speed_limit': self.speed_limit,
            'pixels_per_meter': self.calibration_pixels_per_meter,
            'calibration_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        try:
            with open("speed_config.json", 'w') as f:
                json.dump(config, f, indent=2)
            print(f"Saved calibration: {self.calibration_pixels_per_meter} px/m, limit: {self.speed_limit} km/h")
        except Exception as e:
            print(f"Could not save calibration: {e}")
    
    def calibrate_speed(self, frame, reference_distance_meters=10):
        """
        Interactive calibration - DISABLED for headless operation.
        Uses default calibration values.
        """
        print("\n" + "="*60)
        print("SPEED CALIBRATION")
        print("="*60)
        print("Using default calibration values.")
        print("To calibrate manually, run this script in a local environment with display.")
        return True
    
    def calculate_speed(self, track_id, center_x, center_y, frame_count, fps):
        """
        Calculate speed based on vehicle movement across frames
        Returns speed in km/h if enough data, otherwise None
        """
        if not hasattr(self, 'vehicle_positions'):
            self.vehicle_positions = {}
        
        if track_id not in self.vehicle_positions:
            self.vehicle_positions[track_id] = []
        
        # Store current position
        self.vehicle_positions[track_id].append((frame_count, center_x, center_y))
        
        # Keep only last 30 positions (about 1 second at 30 fps)
        if len(self.vehicle_positions[track_id]) > 30:
            self.vehicle_positions[track_id].pop(0)
        
        # Need at least 2 positions to calculate speed
        if len(self.vehicle_positions[track_id]) < 2:
            return None
        
        # Get first and last position
        first_frame, first_x, first_y = self.vehicle_positions[track_id][0]
        last_frame, last_x, last_y = self.vehicle_positions[track_id][-1]
        
        # Calculate frame difference
        frame_diff = last_frame - first_frame
        if frame_diff == 0:
            return None
        
        # Calculate pixel distance moved
        pixel_distance = ((last_x - first_x) ** 2 + (last_y - first_y) ** 2) ** 0.5
        
        # Convert to meters using calibration
        distance_meters = pixel_distance / self.calibration_pixels_per_meter
        
        # Calculate time in seconds
        time_seconds = frame_diff / fps
        
        # Calculate speed in km/h
        speed_mps = distance_meters / time_seconds if time_seconds > 0 else 0
        speed_kmh = speed_mps * 3.6
        
        return speed_kmh
    
    def load_models(self):
        """Load all models with GPU acceleration"""
        print("\nLoading models...")
        
        self.vehicle_model = YOLO(self.vehicle_model_path)
        self.traffic_light_model = YOLO(self.traffic_light_model_path)
        self.helmet_model = YOLO(self.helmet_model_path)
        
        if DEVICE != 'cpu':
            self.vehicle_model.to('cuda')
            self.traffic_light_model.to('cuda')
            self.helmet_model.to('cuda')
            print("Models moved to GPU")
        
        # Initialize license plate detector
        self.plate_detector = LicensePlateDetector()
        
        print("All models loaded successfully")
    
    def process_video(self, video_path, video_filename, calibrate=False):
        """Process video with all detectors in a single pass using GPU"""
        
        # Create session folder
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(self.results_base, session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        print(f"\nProcessing video: {video_filename}")
        print(f"Session ID: {session_id}")
        print(f"Device: {'GPU' if DEVICE != 'cpu' else 'CPU'}")
        
        # Load models
        self.load_models()
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Read first frame for calibration if needed
        ret, first_frame = cap.read()
        if not ret:
            print("Error: Could not read first frame")
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Calibrate if requested
        if calibrate:
            print("Calibration requested. Using default values.")
        
        # Setup output video writer
        output_video_path = os.path.join(session_dir, f"{video_filename}_analyzed.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Create directories for violation screenshots
        screenshots_dir = os.path.join(session_dir, "screenshots")
        plates_dir = os.path.join(session_dir, "license_plates")
        os.makedirs(screenshots_dir, exist_ok=True)
        os.makedirs(plates_dir, exist_ok=True)
        
        # Processing variables
        frame_count = 0
        current_light_state = None
        light_confidence = 0
        violations = {
            'red_light': [],
            'speeding': [],
            'helmet': []
        }
        
        # Speed tracking
        self.vehicle_positions = {}
        recorded_speeding = set()
        
        # Red light tracking
        recorded_red_light = set()
        spatial_red_light = set()
        X_ZONE_GRID = 160
        
        # Helmet tracking
        recorded_helmet = {}
        HELMET_COOLDOWN = 90
        
        # Stop line position (for red light detection)
        stop_line_y = int(height * 0.7)
        stop_line_x_start = 0
        stop_line_x_end = width
        
        red_light_active = False
        
        print("\nProcessing video frames...")
        print("Press Ctrl+C to stop early")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Progress indicator every 30 frames
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {frame_count}/{total_frames} ({progress:.1f}%)")
                
                # 1. Traffic light detection (every 10 frames to save time)
                if frame_count % 10 == 0:
                    light_results = self.traffic_light_model(frame, conf=0.3, device=DEVICE, verbose=False)
                    for r in light_results:
                        if r.boxes is not None:
                            for box in r.boxes:
                                cls_id = int(box.cls[0])
                                conf = float(box.conf[0])
                                light_state = self.traffic_light_model.names[cls_id]
                                if conf > light_confidence:
                                    light_confidence = conf
                                    current_light_state = light_state
                
                # Check if light is red
                is_red = current_light_state in ['red', 'redyellow', 'unknown']
                
                # Track red light cycle
                if is_red and not red_light_active:
                    red_light_active = True
                elif not is_red and red_light_active:
                    recorded_red_light = set()
                    spatial_red_light = set()
                    red_light_active = False
                
                # 2. Vehicle detection with tracking
                vehicle_results = self.vehicle_model.track(frame, persist=True, device=DEVICE, verbose=False, conf=0.4)
                
                # 3. Helmet detection
                helmet_results = self.helmet_model(frame, conf=0.4, device=DEVICE, verbose=False)
                
                # Annotate frame
                annotated = frame.copy()
                
                # Draw stop line
                cv2.line(annotated, (stop_line_x_start, stop_line_y), (stop_line_x_end, stop_line_y), (0, 255, 0), 2)
                cv2.putText(annotated, "STOP LINE", (stop_line_x_start + 10, stop_line_y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Process vehicle detections for red light and speed violations
                if vehicle_results[0].boxes is not None:
                    boxes = vehicle_results[0].boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        vehicle_type = self.vehicle_model.names[cls_id]
                        
                        # Get track ID
                        track_id = None
                        if hasattr(box, 'id') and box.id is not None:
                            track_id = int(box.id[0])
                        else:
                            center_x = (x1 + x2) // 2
                            track_id = f"pos_{center_x//50}_{frame_count//30}"
                        
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        bottom_y = y2
                        x_zone = center_x // X_ZONE_GRID
                        
                        # ----- SPEED DETECTION -----
                        speed = self.calculate_speed(track_id, center_x, center_y, frame_count, fps)
                        
                        # Check for speeding violation - SAVE TO DATABASE
                        if speed is not None and speed > self.speed_limit:
                            print(f"  [SPEED] Vehicle {track_id} at {speed:.1f} km/h - EXCEEDS LIMIT")
                            if track_id not in recorded_speeding:
                                recorded_speeding.add(track_id)
                                
                                violation_id = len(violations['speeding']) + 1
                                screenshot_path = os.path.join(screenshots_dir, f"speeding_{violation_id}.jpg")
                                
                                # Annotate frame with speed info for screenshot
                                speed_annotated = annotated.copy()
                                cv2.putText(speed_annotated, f"SPEEDING: {speed:.1f} km/h (Limit: {self.speed_limit})", 
                                           (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                cv2.imwrite(screenshot_path, speed_annotated)
                                
                                violation_data = {
                                    'id': violation_id,
                                    'frame': frame_count,
                                    'timestamp': frame_count / fps,
                                    'vehicle_type': vehicle_type,
                                    'confidence': conf,
                                    'track_id': str(track_id),
                                    'speed': round(speed, 1),
                                    'speed_limit': self.speed_limit,
                                    'screenshot': f"speeding_{violation_id}.jpg"
                                }
                                violations['speeding'].append(violation_data)
                                
                                # Save to database
                                save_violation(
                                    session_id,
                                    'speeding',
                                    vehicle_type,
                                    frame_count / fps,
                                    frame_count,
                                    conf,
                                    screenshot_path,
                                    video_filename
                                )
                                print(f"  [DB] Speeding violation saved: {vehicle_type} at {speed:.1f} km/h")
                        
                        # ----- RED LIGHT DETECTION WITH LICENSE PLATE -----
                        if is_red and bottom_y > stop_line_y:
                            if stop_line_x_start <= center_x <= stop_line_x_end:
                                already_by_id = (track_id is not None and track_id in recorded_red_light)
                                already_by_zone = (x_zone in spatial_red_light)
                                
                                if not already_by_id and not already_by_zone:
                                    if track_id is not None:
                                        recorded_red_light.add(track_id)
                                    spatial_red_light.add(x_zone)
                                    
                                    violation_id = len(violations['red_light']) + 1
                                    screenshot_path = os.path.join(screenshots_dir, f"red_light_{violation_id}.jpg")
                                    
                                    # ----- LICENSE PLATE DETECTION -----
                                    plate_text = None
                                    plate_confidence = 0
                                    plate_image_path = None
                                    plate_box = None
                                    
                                    try:
                                        # Process the vehicle to detect license plate
                                        plate_result = self.plate_detector.process_vehicle(frame, (x1, y1, x2, y2))
                                        
                                        if plate_result:
                                            plate_text = plate_result['plate_text']
                                            plate_confidence = plate_result['confidence']
                                            plate_box = plate_result['plate_box']
                                            
                                            # Save cropped plate image
                                            plate_image_path = self.plate_detector.save_plate_image(
                                                frame,
                                                plate_box,
                                                plate_text,
                                                violation_id,
                                                session_dir
                                            )
                                            print(f"  [PLATE] Detected: {plate_text} (conf: {plate_confidence:.2f})")
                                            
                                            # Draw plate bounding box on annotated frame
                                            px1, py1, px2, py2 = plate_box
                                            cv2.rectangle(annotated, (px1, py1), (px2, py2), (255, 255, 0), 2)
                                            cv2.putText(annotated, plate_text, (px1, py1 - 5),
                                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                                        else:
                                            print(f"  [PLATE] No plate detected for vehicle {track_id}")
                                            
                                    except Exception as e:
                                        print(f"  [PLATE] Error: {e}")
                                    
                                    # Save the annotated screenshot
                                    cv2.imwrite(screenshot_path, annotated)
                                    
                                    violation_data = {
                                        'id': violation_id,
                                        'frame': frame_count,
                                        'timestamp': frame_count / fps,
                                        'vehicle_type': vehicle_type,
                                        'confidence': conf,
                                        'track_id': str(track_id),
                                        'screenshot': f"red_light_{violation_id}.jpg",
                                        'license_plate': plate_text,
                                        'plate_confidence': plate_confidence,
                                        'plate_image': plate_image_path
                                    }
                                    violations['red_light'].append(violation_data)
                                    
                                    # Save to database with license plate
                                    save_violation(
                                        session_id,
                                        'red_light',
                                        vehicle_type,
                                        frame_count / fps,
                                        frame_count,
                                        conf,
                                        screenshot_path,
                                        video_filename,
                                        license_plate=plate_text,
                                        plate_confidence=plate_confidence,
                                        plate_image_path=plate_image_path
                                    )
                                    print(f"  [DB] Red light violation saved: {vehicle_type} | Plate: {plate_text if plate_text else 'Not detected'}")
                        
                        # ----- VISUAL DISPLAY (BOUNDING BOX AND LABELS) -----
                        # Determine if this vehicle is currently violating
                        is_speeding_vehicle = (speed is not None and speed > self.speed_limit)
                        is_red_light_violation = (track_id in recorded_red_light)
                        
                        # Set thickness and color based on violation priority
                        if is_speeding_vehicle:
                            thickness = 3
                            color = (0, 0, 255)  # Bright red for speeding
                        elif is_red_light_violation:
                            thickness = 3
                            color = (0, 0, 255)  # Bright red for red light
                        else:
                            thickness = 2
                            color = (0, 255, 0)  # Green for normal
                        
                        # Draw bounding box
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
                        
                        # Build label with speed if available
                        if speed is not None:
                            label = f"{vehicle_type} {conf:.2f} | {speed:.1f} km/h"
                        else:
                            label = f"{vehicle_type} {conf:.2f}"
                        
                        # Add violation label above the vehicle if applicable
                        if is_speeding_vehicle:
                            violation_label = f"SPEEDING: {speed:.1f} km/h (Limit: {self.speed_limit})"
                            (text_w, text_h), baseline = cv2.getTextSize(violation_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(annotated, (x1, y1 - text_h - 25), (x1 + text_w + 10, y1 - 20), (0, 0, 255), -1)
                            cv2.putText(annotated, violation_label, (x1 + 5, y1 - 28), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        elif is_red_light_violation:
                            violation_label = "RED LIGHT VIOLATION"
                            (text_w, text_h), baseline = cv2.getTextSize(violation_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(annotated, (x1, y1 - text_h - 25), (x1 + text_w + 10, y1 - 20), (0, 0, 255), -1)
                            cv2.putText(annotated, violation_label, (x1 + 5, y1 - 28), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Draw vehicle label
                        cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Process helmet detections
                if helmet_results[0].boxes is not None:
                    for box in helmet_results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = self.helmet_model.names[cls_id]
                        
                        rider_zone = (x1 // X_ZONE_GRID, y1 // X_ZONE_GRID)
                        is_no_helmet = ('without_helmet' in class_name.lower() or 'nohelmet' in class_name.lower())
                        
                        if is_no_helmet:
                            already = ((frame_count - recorded_helmet.get(rider_zone, -HELMET_COOLDOWN)) < HELMET_COOLDOWN)
                            if not already:
                                recorded_helmet[rider_zone] = frame_count
                                
                                violation_id = len(violations['helmet']) + 1
                                screenshot_path = os.path.join(screenshots_dir, f"helmet_{violation_id}.jpg")
                                cv2.imwrite(screenshot_path, annotated)
                                
                                violation_data = {
                                    'id': violation_id,
                                    'frame': frame_count,
                                    'timestamp': frame_count / fps,
                                    'class': class_name,
                                    'confidence': conf,
                                    'screenshot': f"helmet_{violation_id}.jpg"
                                }
                                violations['helmet'].append(violation_data)
                                
                                save_violation(
                                    session_id,
                                    'helmet',
                                    class_name,
                                    frame_count / fps,
                                    frame_count,
                                    conf,
                                    screenshot_path,
                                    video_filename
                                )
                                print(f"  [DB] Helmet violation saved: {class_name}")
                            
                            # Draw with thicker red box for violation
                            color = (0, 0, 255)
                            thickness = 3
                            label = f"NO HELMET {conf:.2f}"
                            
                            # Add violation label above
                            violation_label = "HELMET VIOLATION"
                            (text_w, text_h), baseline = cv2.getTextSize(violation_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(annotated, (x1, y1 - text_h - 25), (x1 + text_w + 10, y1 - 20), (0, 0, 255), -1)
                            cv2.putText(annotated, violation_label, (x1 + 5, y1 - 28), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        else:
                            # Green for helmet worn
                            color = (0, 255, 0)
                            thickness = 2
                            label = f"HELMET {conf:.2f}"
                        
                        # Draw bounding box
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
                        cv2.putText(annotated, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Add traffic light state to frame
                if current_light_state:
                    light_color = (0, 0, 255) if is_red else (0, 255, 0)
                    cv2.putText(annotated, f"TRAFFIC LIGHT: {current_light_state.upper()} ({light_confidence:.2f})", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, light_color, 2)
                
                # Add speed limit to frame
                cv2.putText(annotated, f"Speed Limit: {self.speed_limit} km/h", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Add frame counter and violation counts
                cv2.putText(annotated, f"Frame: {frame_count}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(annotated, f"Violations: RL:{len(violations['red_light'])} Spd:{len(violations['speeding'])} Hlm:{len(violations['helmet'])}", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Draw legend on the bottom right corner
                legend_y = height - 80
                legend_x = width - 180
                
                # Background for legend
                cv2.rectangle(annotated, (legend_x - 10, legend_y - 70), (width - 10, height - 10), (0, 0, 0), -1)
                cv2.rectangle(annotated, (legend_x - 10, legend_y - 70), (width - 10, height - 10), (100, 100, 100), 1)
                
                # Legend title
                cv2.putText(annotated, "LEGEND", (legend_x, legend_y - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Normal vehicle
                cv2.rectangle(annotated, (legend_x, legend_y - 40), (legend_x + 20, legend_y - 25), (0, 255, 0), 2)
                cv2.putText(annotated, "Normal", (legend_x + 25, legend_y - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Speeding violation
                cv2.rectangle(annotated, (legend_x, legend_y - 20), (legend_x + 20, legend_y - 5), (0, 0, 255), 3)
                cv2.putText(annotated, "Speeding", (legend_x + 25, legend_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Red light violation
                cv2.rectangle(annotated, (legend_x, legend_y - 0), (legend_x + 20, legend_y + 15), (0, 0, 255), 3)
                cv2.putText(annotated, "Red Light", (legend_x + 25, legend_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Helmet violation
                cv2.rectangle(annotated, (legend_x, legend_y + 20), (legend_x + 20, legend_y + 35), (0, 0, 255), 3)
                cv2.putText(annotated, "No Helmet", (legend_x + 25, legend_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Write frame to output video
                out.write(annotated)
                
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
        
        # Cleanup - GUI calls removed for headless operation
        cap.release()
        out.release()
        # cv2.destroyAllWindows() - DISABLED for headless environment

        # Post-processing dedup
        violations['red_light'] = _dedup_violations(violations['red_light'], frame_window=30, x_tolerance=160)
        for i, v in enumerate(violations['red_light'], 1):
            v['id'] = i
        
        violations['speeding'] = _dedup_violations(violations['speeding'], frame_window=30, x_tolerance=160)
        for i, v in enumerate(violations['speeding'], 1):
            v['id'] = i
        
        violations['helmet'] = _dedup_violations(violations['helmet'], frame_window=30, x_tolerance=160)
        for i, v in enumerate(violations['helmet'], 1):
            v['id'] = i
        
        # Calculate total vehicles from tracked IDs
        total_vehicles = len(set([v.get('track_id') for v in violations['red_light']] + 
                                 [v.get('track_id') for v in violations['speeding']] + 
                                 [str(i) for i in range(50)]))  # Approximate

        # Prepare results
        results = {
            'session_id': session_id,
            'video_filename': video_filename,
            'video_path': video_path,
            'processed_video': output_video_path,
            'processing_time': frame_count / fps if fps > 0 else 0,
            'duration': frame_count / fps if fps > 0 else 0,
            'total_frames': frame_count,
            'violations': violations,
            'summary': {
                'total_violations': len(violations['red_light']) + len(violations['speeding']) + len(violations['helmet']),
                'red_light_count': len(violations['red_light']),
                'speeding_count': len(violations['speeding']),
                'helmet_count': len(violations['helmet']),
                'total_vehicles': total_vehicles
            }
        }
        
        # Save final report
        report_path = os.path.join(session_dir, 'final_report.json')
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"Total frames processed: {frame_count}")
        print(f"Total violations: {results['summary']['total_violations']}")
        print(f"  - Red light violations: {len(violations['red_light'])}")
        print(f"  - Speeding violations: {len(violations['speeding'])}")
        print(f"  - Helmet violations: {len(violations['helmet'])}")
        print(f"\nSpeed calibration: {self.calibration_pixels_per_meter:.2f} px/m")
        print(f"Speed limit: {self.speed_limit} km/h")
        print(f"\nViolations saved to database")
        print(f"Results saved to: {session_dir}")
        
        return results

def process_uploaded_video(video_path, video_filename, calibrate=False):
    system = TrafficMonitoringSystem()
    return system.process_video(video_path, video_filename, calibrate)

if __name__ == "__main__":
    print("main_processor.py loaded successfully")