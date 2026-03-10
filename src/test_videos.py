<<<<<<< HEAD
=======
#!/usr/bin/env python3
"""
Traffic Video Vehicle Detection with Confidence Scores
WITH SCREEN FITTING - Videos will fit to screen
"""
>>>>>>> c1c1127 (Initial commit)
from ultralytics import YOLO
import cv2
import os
import sys
import time
import argparse
import numpy as np

<<<<<<< HEAD
# Import the speed detection modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from speed_detector import SpeedDetector 
from vehicle_tracker import SimpleTracker

class TrafficVideoDetector:
    def __init__(self, model_path="models/yolov8n_quick_test/weights/best.pt"):
        """Initialize the detector with trained model"""
        print(" Initializing Traffic Video Detector...")
        
        # Check if model exists
        if not os.path.exists(model_path):
            print(f" ERROR: Model not found at {model_path}")
=======
class TrafficVideoDetector:
    def __init__(self, model_path="models/yolov8n_quick_test/weights/best.pt"):
        """Initialize the detector with trained model"""
        print("🚗 Initializing Traffic Video Detector...")
        
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"❌ ERROR: Model not found at {model_path}")
>>>>>>> c1c1127 (Initial commit)
            print("Please check the path or train the model first")
            print(f"Current directory: {os.getcwd()}")
            print("\nLooking for alternative models...")
            
            # Try to find any .pt file
            possible_paths = [
                "models/yolov8n_quick_test/weights/best.pt",
                "models/yolov8n_ua_detrac/weights/best.pt",
                "models/train/weights/best.pt",
                "runs/detect/train/weights/best.pt",
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
<<<<<<< HEAD
                    print(f" Found alternative model: {model_path}")
                    break
            else:
                print(" No trained models found!")
=======
                    print(f"✅ Found alternative model: {model_path}")
                    break
            else:
                print("❌ No trained models found!")
>>>>>>> c1c1127 (Initial commit)
                print("Run training first: python src/03_train.py")
                sys.exit(1)
        
        # Load the trained model
<<<<<<< HEAD
        print(f" Loading model: {model_path}")
        try:
            self.model = YOLO(model_path)
            print(f" Model loaded successfully!")
        except Exception as e:
            print(f" Failed to load model: {e}")
=======
        print(f"📦 Loading model: {model_path}")
        try:
            self.model = YOLO(model_path)
            print(f"✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
>>>>>>> c1c1127 (Initial commit)
            sys.exit(1)
        
        # Get class names
        self.class_names = self.model.names if hasattr(self.model, 'names') else {}
<<<<<<< HEAD
        print(f" Detecting {len(self.class_names)} classes: {list(self.class_names.values())}")
        
        # Initialize tracker and speed detector
        print(" Initializing vehicle tracker and speed detector...")
        self.tracker = SimpleTracker()
        self.speed_detector = SpeedDetector()
        
        # Colors for different classes (with speed-based coloring)
        self.class_colors = {
            'car': (0, 255, 0),      # Green (default)
            'bus': (0, 255, 0),      # Blue
            'truck': (0, 255, 0),    # Red
            'van': (0, 255, 0),    # Cyan
        }
        
        # Speed settings
        self.speed_limit = 50  # Default speed limit in km/h
        self.calibration_set = False
        
=======
        print(f"🎯 Detecting {len(self.class_names)} classes: {list(self.class_names.values())}")
        
        # Colors for different classes
        self.colors = {
            'car': (0, 255, 0),      # Green
            'bus': (255, 0, 0),      # Blue
            'truck': (0, 0, 255),    # Red
            'van': (255, 255, 0),    # Cyan
        }
        
>>>>>>> c1c1127 (Initial commit)
        # Statistics
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'detections_by_class': {},
            'processing_time': 0,
<<<<<<< HEAD
            'average_confidence': 0,
            'speed_violations': 0,
            'max_speed': 0,
            'average_speed': 0,
        }
        
        # Speed data storage
        self.speeds_history = []
    
    def set_calibration(self, known_distance_meters, pixel_distance, speed_limit=None):
        """
        Calibrate the camera for accurate speed detection
        known_distance_meters: Real distance in meters (e.g., lane width = 3.5m)
        pixel_distance: Distance in pixels from video frame
        """
        print(f"\n Calibrating speed detector...")
        print(f"   Known distance: {known_distance_meters} meters")
        print(f"   Pixel distance: {pixel_distance} pixels")
        
        calibration = self.speed_detector.calibrate_camera(known_distance_meters, pixel_distance)
        
        if speed_limit:
            self.speed_limit = speed_limit
            self.speed_detector.calibration['speed_limit'] = speed_limit
        
        self.calibration_set = True
        print(f"   Calibration complete: {calibration['pixels_per_meter']:.2f} pixels/meter")
        print(f"   Speed limit set to: {self.speed_limit} km/h")
        
        # Save calibration to file
        calibration_dir = "calibration_data"
        os.makedirs(calibration_dir, exist_ok=True)
        calibration_file = os.path.join(calibration_dir, "default_calibration.json")
        
        import json
        with open(calibration_file, 'w') as f:
            json.dump(calibration, f, indent=2)
        
        print(f"   Calibration saved to: {calibration_file}")
        return calibration
    
    def _resize_to_fit_screen(self, frame, max_width=1580, max_height=800):
=======
            'average_confidence': 0
        }
    
    def _resize_to_fit_screen(self, frame, max_width=1280, max_height=720):
>>>>>>> c1c1127 (Initial commit)
        """Resize frame to fit screen while maintaining aspect ratio"""
        height, width = frame.shape[:2]
        
        # Calculate scaling factor
        scale_width = max_width / width
        scale_height = max_height / height
        scale = min(scale_width, scale_height)
        
        # If image is already smaller than max, don't scale up
        if scale > 1:
            return frame
        
        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized
    
    def process_video(self, video_path, output_path=None, show_display=True, save_output=True):
<<<<<<< HEAD
        """Process a video file and detect vehicles WITH SPEED"""
        
        if not os.path.exists(video_path):
            print(f" Video not found: {video_path}")
            print("Please provide a valid video file path")
            return False
        
        print(f"\n Processing video: {video_path}")
        
        # Ask for calibration if not set
        if not self.calibration_set:
            print("\n  SPEED DETECTION CALIBRATION REQUIRED ")
            print("For accurate speed measurement, you need to calibrate the camera.")
            print("Example: If you know a lane is 3.5 meters wide, measure it in pixels.")
            
            calibrate_now = input("Do you want to calibrate now? (y/n): ").strip().lower()
            
            if calibrate_now == 'y':
                try:
                    known_dist = float(input("Enter known distance in meters (e.g., 3.5 for lane width): "))
                    pixel_dist = float(input("Enter that distance in pixels from video: "))
                    speed_lim = input(f"Enter speed limit in km/h [default {self.speed_limit}]: ").strip()
                    
                    if speed_lim:
                        self.speed_limit = float(speed_lim)
                    
                    self.set_calibration(known_dist, pixel_dist, self.speed_limit)
                except ValueError:
                    print("Invalid input. Using default calibration (may be inaccurate).")
            else:
                print("Using default calibration (may be inaccurate for speed).")
=======
        """Process a video file and detect vehicles"""
        
        if not os.path.exists(video_path):
            print(f"❌ Video not found: {video_path}")
            print("Please provide a valid video file path")
            return False
        
        print(f"\n📹 Processing video: {video_path}")
>>>>>>> c1c1127 (Initial commit)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
<<<<<<< HEAD
            print(" Could not open video file")
=======
            print("❌ Could not open video file")
>>>>>>> c1c1127 (Initial commit)
            return False
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
<<<<<<< HEAD
        if fps == 0:
            fps = 30  # Default if can't detect
            print(f" Warning: Could not detect FPS. Using default: {fps}")
        
=======
>>>>>>> c1c1127 (Initial commit)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
<<<<<<< HEAD
        print(f"    Video Info: {width}x{height}, {fps} FPS, {total_frames} frames")
        print(f"    Speed Limit: {self.speed_limit} km/h")
        print(f"    Estimated time: {total_frames/fps/60:.1f} minutes on CPU")
=======
        print(f"   📊 Video Info: {width}x{height}, {fps} FPS, {total_frames} frames")
        print(f"   ⏳ Estimated time: {total_frames/fps/60:.1f} minutes on CPU")
>>>>>>> c1c1127 (Initial commit)
        
        # Setup output video if saving (original size)
        if save_output:
            if output_path is None:
                # Create a clean output filename
                base_name = os.path.basename(video_path)
                name_without_ext = os.path.splitext(base_name)[0]
<<<<<<< HEAD
                output_path = f"outputs/processed_{name_without_ext}_with_speed.mp4"
=======
                output_path = f"outputs/processed_{name_without_ext}.mp4"
>>>>>>> c1c1127 (Initial commit)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
<<<<<<< HEAD
            print(f"   Output will be saved to: {output_path}")
=======
            print(f"   💾 Output will be saved to: {output_path}")
>>>>>>> c1c1127 (Initial commit)
        else:
            out = None
        
        # Reset statistics
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'detections_by_class': {},
            'processing_time': 0,
<<<<<<< HEAD
            'average_confidence': 0,
            'speed_violations': 0,
            'max_speed': 0,
            'average_speed': 0,
        }
        self.speeds_history = []
        
        confidences = []
        speeds_list = []  # Store all speeds for statistics
        start_time = time.time()
        frame_count = 0
        
        print("\n Processing frames...")
        print("   (Press 'q' to stop early, 'p' to pause, 's' to save screenshot)")
        print("   (Press 'c' to show/hide speed calibration info)")
        
        paused = False
        show_calibration_info = True
=======
            'average_confidence': 0
        }
        
        confidences = []
        start_time = time.time()
        frame_count = 0
        
        print("\n⏳ Processing frames...")
        print("   (Press 'q' to stop early, 'p' to pause, 's' to save screenshot)")
        
        paused = False
>>>>>>> c1c1127 (Initial commit)
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                self.stats['total_frames'] += 1
                
                # Display progress
                if frame_count % 50 == 0:
                    progress = (frame_count / total_frames) * 100
                    elapsed = time.time() - start_time
                    fps_current = frame_count / elapsed if elapsed > 0 else 0
                    print(f"   Processed: {frame_count}/{total_frames} frames ({progress:.1f}%) - {fps_current:.1f} FPS")
                
                # Run detection
                results = self.model(frame, device='cpu', verbose=False, conf=0.25)
                
                # Process detections
                detections_this_frame = 0
                frame_confidences = []
<<<<<<< HEAD
                frame_speeds = []
                
                # Extract detections for tracking
                raw_detections = []
=======
                
>>>>>>> c1c1127 (Initial commit)
                for result in results:
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        boxes = result.boxes
                        
                        for box in boxes:
                            # Get box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            # Get class and confidence
                            cls_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            
<<<<<<< HEAD
                            raw_detections.append([x1, y1, x2, y2, confidence, cls_id])
                
                # Track vehicles across frames
                tracked_detections = self.tracker.update(raw_detections)
                
                # Update speed detector with tracked vehicles
                self.speed_detector.update_tracks(tracked_detections, frame_count)
                
                # Calculate speeds for all tracked vehicles
                speeds = self.speed_detector.get_all_speeds(fps)
                
                # Process each tracked detection
                for det in tracked_detections:
                    if len(det) >= 7:  # Has track_id
                        x1, y1, x2, y2, confidence, cls_id, track_id = det
                        
                        # Get class name
                        class_name = self.class_names.get(cls_id, f"Class {cls_id}")
                        
                        # Update statistics
                        detections_this_frame += 1
                        frame_confidences.append(confidence)
                        self.stats['detections_by_class'][class_name] = self.stats['detections_by_class'].get(class_name, 0) + 1
                        
                        # Get speed for this vehicle
                        speed = speeds.get(track_id, 0)
                        if speed > 0:
                            frame_speeds.append(speed)
                            speeds_list.append(speed)
                            
                            # Update max speed
                            if speed > self.stats['max_speed']:
                                self.stats['max_speed'] = speed
                        
                        # Check if speeding
                        is_speeding = speed > self.speed_limit
                        if is_speeding:
                            self.stats['speed_violations'] += 1
                        
                        # Get color based on speed
                        if is_speeding:
                            color = (0, 0, 255)  # Red for speeding
                        else:
                            color = self.class_colors.get(class_name, (255, 255, 255))
                        
                        # Draw bounding box (thicker for speeding)
                        thickness = 4 if is_speeding else 2
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                        
                        # Create label with class, speed, and track ID
                        if speed > 0:
                            label = f"{class_name} ID:{track_id} {speed:.1f}km/h"
                        else:
                            label = f"{class_name} ID:{track_id}"
                        
                        # Add speed warning if speeding
                        if is_speeding:
                            label += " ⚠️"
                        
                        # Calculate label size
                        (label_width, label_height), baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                        )
                        
                        # Draw label background
                        cv2.rectangle(frame, 
                                     (x1, y1 - label_height - 10),
                                     (x1 + label_width, y1),
                                     color, -1)
                        
                        # Draw label text
                        cv2.putText(frame, label,
                                   (x1, y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX,
                                   0.5, (255, 255, 255), 2)
                
                # Add frame counter and stats overlay
                self._add_overlay(frame, frame_count, detections_this_frame, 
                                 np.mean(frame_confidences) if frame_confidences else 0,
                                 np.mean(frame_speeds) if frame_speeds else 0,
                                 show_calibration_info)
=======
                            # Get class name
                            class_name = self.class_names.get(cls_id, f"Class {cls_id}")
                            
                            # Update statistics
                            detections_this_frame += 1
                            frame_confidences.append(confidence)
                            self.stats['detections_by_class'][class_name] = self.stats['detections_by_class'].get(class_name, 0) + 1
                            
                            # Get color for this class
                            color = self.colors.get(class_name, (255, 255, 255))
                            
                            # Draw bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            
                            # Create label with class and confidence
                            label = f"{class_name}: {confidence:.2f}"
                            
                            # Calculate label size
                            (label_width, label_height), baseline = cv2.getTextSize(
                                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                            )
                            
                            # Draw label background
                            cv2.rectangle(frame, 
                                         (x1, y1 - label_height - 10),
                                         (x1 + label_width, y1),
                                         color, -1)
                            
                            # Draw label text
                            cv2.putText(frame, label,
                                       (x1, y1 - 5),
                                       cv2.FONT_HERSHEY_SIMPLEX,
                                       0.5, (255, 255, 255), 2)
                
                # Add frame counter and stats overlay
                self._add_overlay(frame, frame_count, detections_this_frame, 
                                 np.mean(frame_confidences) if frame_confidences else 0)
>>>>>>> c1c1127 (Initial commit)
                
                # Add to statistics
                if frame_confidences:
                    confidences.extend(frame_confidences)
                    self.stats['total_detections'] += detections_this_frame
                
                # Save frame if requested (save in original size)
                if save_output and out is not None:
                    out.write(frame)
                
                # Display if requested (show resized to fit screen)
                if show_display:
                    display_frame = self._resize_to_fit_screen(frame)
<<<<<<< HEAD
                    cv2.imshow('Traffic Detection with Speed - Press q to quit', display_frame)
=======
                    cv2.imshow('Traffic Detection - Press q to quit', display_frame)
>>>>>>> c1c1127 (Initial commit)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
<<<<<<< HEAD
                print("\n  Processing stopped by user")
=======
                print("\n⚠️  Processing stopped by user")
>>>>>>> c1c1127 (Initial commit)
                break
            elif key == ord('p'):
                paused = not paused
                if paused:
<<<<<<< HEAD
                    print("  Paused. Press 'p' to resume")
                else:
                    print("  Resumed")
=======
                    print("⏸️  Paused. Press 'p' to resume")
                else:
                    print("▶️  Resumed")
>>>>>>> c1c1127 (Initial commit)
            elif key == ord('s'):
                # Save screenshot
                screenshot_dir = "outputs/screenshots"
                os.makedirs(screenshot_dir, exist_ok=True)
<<<<<<< HEAD
                screenshot_path = f"{screenshot_dir}/frame_{frame_count:06d}_speed.jpg"
                cv2.imwrite(screenshot_path, frame)
                print(f" Screenshot saved: {screenshot_path}")
            elif key == ord('c'):
                show_calibration_info = not show_calibration_info
                print(f"  Calibration info: {'SHOWN' if show_calibration_info else 'HIDDEN'}")
            elif key == ord('v'):
                # Save current speeds to file
                speeds_file = "outputs/current_speeds.csv"
                with open(speeds_file, 'w') as f:
                    f.write("track_id,speed_kmh,class,frame\n")
                    for track_id, speed in speeds.items():
                        f.write(f"{track_id},{speed:.2f},vehicle,{frame_count}\n")
                print(f"  Current speeds saved to: {speeds_file}")
=======
                screenshot_path = f"{screenshot_dir}/frame_{frame_count:06d}.jpg"
                cv2.imwrite(screenshot_path, frame)
                print(f"📸 Screenshot saved: {screenshot_path}")
            elif key == ord('f'):
                # Toggle fullscreen (if supported)
                print("🖥️  Fullscreen toggle not implemented in OpenCV")
>>>>>>> c1c1127 (Initial commit)
        
        # Cleanup
        processing_time = time.time() - start_time
        self.stats['processing_time'] = processing_time
        
        if confidences:
            self.stats['average_confidence'] = np.mean(confidences)
        
<<<<<<< HEAD
        if speeds_list:
            self.stats['average_speed'] = np.mean(speeds_list)
        
        cap.release()
        if save_output and out is not None:
            out.release()
            print(f"\n Output video saved: {output_path}")
=======
        cap.release()
        if save_output and out is not None:
            out.release()
            print(f"\n✅ Output video saved: {output_path}")
>>>>>>> c1c1127 (Initial commit)
            print(f"   File size: {os.path.getsize(output_path)//1024} KB")
        
        if show_display:
            cv2.destroyAllWindows()
        
        return True
    
<<<<<<< HEAD
    def _add_overlay(self, frame, frame_num, detections, avg_confidence, avg_speed, show_calibration=True):
        """Add information overlay to frame"""
        # Top left: Frame counter and basic info
        cv2.putText(frame, f"Frame: {frame_num}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Detections: {detections}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Speed info
        if avg_speed > 0:
            cv2.putText(frame, f"Avg Speed: {avg_speed:.1f} km/h", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw speed limit indicator
            cv2.putText(frame, f"Limit: {self.speed_limit} km/h", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Calibration info (toggle with 'c' key)
        if show_calibration and hasattr(self.speed_detector, 'calibration'):
            cal = self.speed_detector.calibration
            if 'pixels_per_meter' in cal:
                cv2.putText(frame, f"Calibration: {cal['pixels_per_meter']:.1f} px/m", 
                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Legend at bottom
        y_offset = frame.shape[0] - 150
        cv2.putText(frame, "LEGEND:", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add speed legend
        y_offset += 30
        cv2.rectangle(frame, (5, y_offset - 15), (25, y_offset), (0, 255, 0), -1)
        cv2.putText(frame, "Normal speed", (30, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += 30
        cv2.rectangle(frame, (5, y_offset - 15), (25, y_offset), (0, 0, 255), -1)
        cv2.putText(frame, "SPEEDING", (30, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add vehicle type legend
        y_offset += 30
        for i, (class_name, color) in enumerate(self.class_colors.items()):
            y_pos = y_offset + (i * 30)
            cv2.rectangle(frame, (5, y_pos - 15), (25, y_pos), color, -1)
=======
    def _add_overlay(self, frame, frame_num, detections, avg_confidence):
        """Add information overlay to frame"""
        # Top left: Frame counter
        cv2.putText(frame, f"Frame: {frame_num}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Top left: Detections counter
        cv2.putText(frame, f"Detections: {detections}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Top left: Average confidence
        if avg_confidence > 0:
            cv2.putText(frame, f"Avg Conf: {avg_confidence:.2f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Bottom left: Legend
        y_offset = frame.shape[0] - 120
        cv2.putText(frame, "LEGEND:", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        for i, (class_name, color) in enumerate(self.colors.items()):
            y_pos = y_offset + 30 + (i * 30)
            cv2.rectangle(frame, (5, y_pos - 20), (25, y_pos - 5), color, -1)
>>>>>>> c1c1127 (Initial commit)
            cv2.putText(frame, f"{class_name}", (30, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def print_statistics(self):
<<<<<<< HEAD
        """Print detailed statistics including speed data"""
        print("\n" + "=" * 70)
        print(" TRAFFIC ANALYSIS WITH SPEED DETECTION - STATISTICS")
        print("=" * 70)
        
        print(f"\n Video Analysis:")
=======
        """Print detailed statistics"""
        print("\n" + "=" * 60)
        print("📊 DETECTION STATISTICS")
        print("=" * 60)
        
        print(f"\n📹 Video Analysis:")
>>>>>>> c1c1127 (Initial commit)
        print(f"   Total frames processed: {self.stats['total_frames']}")
        print(f"   Processing time: {self.stats['processing_time']:.2f} seconds")
        if self.stats['processing_time'] > 0:
            print(f"   Average FPS: {self.stats['total_frames']/self.stats['processing_time']:.1f}")
        
<<<<<<< HEAD
        print(f"\n Vehicle Detection:")
=======
        print(f"\n🚗 Vehicle Detection:")
>>>>>>> c1c1127 (Initial commit)
        print(f"   Total detections: {self.stats['total_detections']}")
        print(f"   Average confidence: {self.stats['average_confidence']:.3f}")
        
        if self.stats['total_frames'] > 0:
            print(f"   Detections per frame: {self.stats['total_detections']/self.stats['total_frames']:.2f}")
        
<<<<<<< HEAD
        print(f"\n SPEED ANALYSIS:")
        print(f"   Speed limit: {self.speed_limit} km/h")
        print(f"   Speed violations detected: {self.stats['speed_violations']}")
        if self.stats['max_speed'] > 0:
            print(f"   Maximum speed detected: {self.stats['max_speed']:.1f} km/h")
            print(f"   Average speed detected: {self.stats['average_speed']:.1f} km/h")
        
        print(f"\n Detections by Vehicle Type:")
=======
        print(f"\n📈 Detections by Class:")
>>>>>>> c1c1127 (Initial commit)
        total = sum(self.stats['detections_by_class'].values())
        if total > 0:
            for class_name, count in sorted(self.stats['detections_by_class'].items(), 
                                           key=lambda x: x[1], reverse=True):
                percentage = (count / total * 100)
                bar = "█" * int(percentage / 2)
                print(f"   {class_name:10s} {count:5d} ({percentage:5.1f}%) {bar}")
        else:
            print("   No detections found")
        
<<<<<<< HEAD
        # Print calibration info
        if hasattr(self.speed_detector, 'calibration'):
            cal = self.speed_detector.calibration
            print(f"\n Calibration Information:")
            print(f"   Pixels per meter: {cal.get('pixels_per_meter', 'Not set'):.2f}")
            print(f"   Assumed FPS: {cal.get('fps', 'Not set')}")
        
        print("\n" + "=" * 70)
=======
        print("\n" + "=" * 60)
>>>>>>> c1c1127 (Initial commit)

def create_sample_video():
    """Create a sample video from validation images"""
    import glob
    
    val_dir = "data/ua_detrac/valid/images"
    if not os.path.exists(val_dir):
<<<<<<< HEAD
        print(f" Validation directory not found: {val_dir}")
=======
        print(f"❌ Validation directory not found: {val_dir}")
>>>>>>> c1c1127 (Initial commit)
        return None
    
    images = sorted(glob.glob(os.path.join(val_dir, "*.jpg")))[:20]
    
    if not images:
<<<<<<< HEAD
        print(" No validation images found")
=======
        print("❌ No validation images found")
>>>>>>> c1c1127 (Initial commit)
        return None
    
    # Read first image for dimensions
    img = cv2.imread(images[0])
    if img is None:
<<<<<<< HEAD
        print(" Could not read image")
=======
        print("❌ Could not read image")
>>>>>>> c1c1127 (Initial commit)
        return None
    
    height, width = img.shape[:2]
    
    # Create video
    sample_video = "outputs/sample_traffic.mp4"
    os.makedirs("outputs", exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(sample_video, fourcc, 5.0, (width, height))
    
<<<<<<< HEAD
    print(f" Creating sample video from {len(images)} images...")
=======
    print(f"📹 Creating sample video from {len(images)} images...")
>>>>>>> c1c1127 (Initial commit)
    for img_path in images:
        img = cv2.imread(img_path)
        if img is not None:
            out.write(img)
    
    out.release()
<<<<<<< HEAD
    print(f" Created sample video: {sample_video}")
=======
    print(f"✅ Created sample video: {sample_video}")
>>>>>>> c1c1127 (Initial commit)
    print(f"   Size: {width}x{height}, {len(images)} frames")
    
    return sample_video

def main():
<<<<<<< HEAD
    parser = argparse.ArgumentParser(description="Traffic Video Vehicle Detection with SPEED MEASUREMENT")
=======
    parser = argparse.ArgumentParser(description="Traffic Video Vehicle Detection with Screen Fitting")
>>>>>>> c1c1127 (Initial commit)
    parser.add_argument("--video", type=str, help="Path to input video file")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    parser.add_argument("--model", type=str, default="models/yolov8n_quick_test/weights/best.pt",
                       help="Path to trained model")
    parser.add_argument("--no-display", action="store_true", help="Don't show video display")
    parser.add_argument("--no-save", action="store_true", help="Don't save output video")
    parser.add_argument("--upload", action="store_true", help="Upload video through interface")
    parser.add_argument("--max-width", type=int, default=1280, help="Maximum display width")
    parser.add_argument("--max-height", type=int, default=720, help="Maximum display height")
<<<<<<< HEAD
    parser.add_argument("--speed-limit", type=int, default=50, help="Speed limit in km/h")
    parser.add_argument("--calibrate", action="store_true", help="Run calibration before processing")
=======
>>>>>>> c1c1127 (Initial commit)
    
    args = parser.parse_args()
    
    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)
    
    # Initialize detector
    detector = TrafficVideoDetector(args.model)
<<<<<<< HEAD
    detector.speed_limit = args.speed_limit
=======
>>>>>>> c1c1127 (Initial commit)
    
    # Add screen size parameters to detector
    detector.max_width = args.max_width
    detector.max_height = args.max_height
    
<<<<<<< HEAD
    # Handle calibration if requested
    if args.calibrate:
        print("\n" + "=" * 60)
        print(" CAMERA CALIBRATION FOR SPEED DETECTION")
        print("=" * 60)
        
        print("\nFor accurate speed measurement, you need to calibrate.")
        print("Example: Measure a known distance (like lane width) in the video.")
        print("Typical lane width in Nepal: 3.5 meters")
        
        try:
            known_dist = float(input("Enter known distance in meters (e.g., 3.5): "))
            pixel_dist = float(input("Enter that distance in pixels from a video frame: "))
            speed_lim = input(f"Enter speed limit in km/h [default {args.speed_limit}]: ").strip()
            
            if speed_lim:
                detector.speed_limit = float(speed_lim)
            
            detector.set_calibration(known_dist, pixel_dist, detector.speed_limit)
        except ValueError:
            print("Invalid input. Using default calibration.")
    
    # Handle upload mode
    if args.upload:
        print("\n Video Upload Interface with Speed Detection")
        print("-" * 60)
=======
    # Handle upload mode
    if args.upload:
        print("\n📤 Video Upload Interface")
        print("-" * 40)
>>>>>>> c1c1127 (Initial commit)
        
        while True:
            video_path = input("Enter path to your traffic video (or 'q' to quit): ").strip()
            
            if video_path.lower() == 'q':
                return
            
            if os.path.exists(video_path):
                break
            else:
<<<<<<< HEAD
                print(f" File not found: {video_path}")
                print("Please enter a valid path")
        
        # Ask for speed limit
        speed_limit_input = input(f"Enter speed limit in km/h [default {detector.speed_limit}]: ").strip()
        if speed_limit_input:
            detector.speed_limit = float(speed_limit_input)
        
        # Ask for output name
        default_output = f"outputs/processed_with_speed_{os.path.basename(video_path)}"
=======
                print(f"❌ File not found: {video_path}")
                print("Please enter a valid path")
        
        # Ask for output name
        default_output = f"outputs/processed_{os.path.basename(video_path)}"
>>>>>>> c1c1127 (Initial commit)
        output_choice = input(f"Output path [default: {default_output}]: ").strip()
        output_path = output_choice if output_choice else default_output
        
        show_display = input("Show live display? (y/n) [default: y]: ").strip().lower() != 'n'
        save_output = input("Save output video? (y/n) [default: y]: ").strip().lower() != 'n'
        
        # Process video
        success = detector.process_video(
            video_path=video_path,
            output_path=output_path,
            show_display=show_display,
            save_output=save_output
        )
    
    # Handle direct video argument
    elif args.video:
        success = detector.process_video(
            video_path=args.video,
            output_path=args.output,
            show_display=not args.no_display,
            save_output=not args.no_save
        )
    
    # Interactive mode
    else:
        print("\n" + "=" * 60)
<<<<<<< HEAD
        print(" TRAFFIC VIDEO DETECTION WITH SPEED MEASUREMENT")
=======
        print("🚗 TRAFFIC VIDEO DETECTION SYSTEM")
>>>>>>> c1c1127 (Initial commit)
        print("=" * 60)
        
        print("\nChoose an option:")
        print("1. Create and test with sample video (automatic)")
        print("2. Upload your own traffic video")
        print("3. Test on video in data/test_videos/")
<<<<<<< HEAD
        print("4. Quick test with calibration")
=======
        print("4. Quick test (single image)")
>>>>>>> c1c1127 (Initial commit)
        
        choice = input("\nEnter choice (1/2/3/4): ").strip()
        
        if choice == "1":
            # Create and test sample video
            sample_video = create_sample_video()
            if sample_video:
                success = detector.process_video(
                    video_path=sample_video,
<<<<<<< HEAD
                    output_path="outputs/processed_sample_with_speed.mp4",
=======
                    output_path="outputs/processed_sample.mp4",
>>>>>>> c1c1127 (Initial commit)
                    show_display=True,
                    save_output=True
                )
            else:
                success = False
        
        elif choice == "2":
<<<<<<< HEAD
            print("\n Upload your traffic video")
            video_path = input("Enter full path to your video file: ").strip()
            
            if os.path.exists(video_path):
                # Ask for speed limit
                speed_limit_input = input(f"Enter speed limit in km/h [default {detector.speed_limit}]: ").strip()
                if speed_limit_input:
                    detector.speed_limit = float(speed_limit_input)
                
                success = detector.process_video(
                    video_path=video_path,
                    output_path=f"outputs/processed_with_speed_{os.path.basename(video_path)}",
=======
            print("\n📤 Upload your traffic video")
            video_path = input("Enter full path to your video file: ").strip()
            
            if os.path.exists(video_path):
                success = detector.process_video(
                    video_path=video_path,
                    output_path=f"outputs/processed_{os.path.basename(video_path)}",
>>>>>>> c1c1127 (Initial commit)
                    show_display=True,
                    save_output=True
                )
            else:
<<<<<<< HEAD
                print(f" File not found: {video_path}")
=======
                print(f"❌ File not found: {video_path}")
>>>>>>> c1c1127 (Initial commit)
                success = False
        
        elif choice == "3":
            # Check for test videos
            test_videos_dir = "data/test_videos"
            if os.path.exists(test_videos_dir):
                videos = [f for f in os.listdir(test_videos_dir) 
                         if f.endswith(('.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI'))]
                
                if videos:
                    print(f"\nFound {len(videos)} videos:")
                    for i, video in enumerate(videos, 1):
                        print(f"  {i}. {video}")
                    
                    try:
                        video_choice = int(input("\nSelect video number: ").strip()) - 1
                        selected_video = videos[video_choice]
                        video_path = os.path.join(test_videos_dir, selected_video)
                        
<<<<<<< HEAD
                        # Ask for speed limit
                        speed_limit_input = input(f"Enter speed limit in km/h [default {detector.speed_limit}]: ").strip()
                        if speed_limit_input:
                            detector.speed_limit = float(speed_limit_input)
                        
                        success = detector.process_video(
                            video_path=video_path,
                            output_path=f"outputs/processed_with_speed_{selected_video}",
=======
                        success = detector.process_video(
                            video_path=video_path,
                            output_path=f"outputs/processed_{selected_video}",
>>>>>>> c1c1127 (Initial commit)
                            show_display=True,
                            save_output=True
                        )
                    except (ValueError, IndexError):
<<<<<<< HEAD
                        print(" Invalid selection")
                        success = False
                else:
                    print(f" No videos found in {test_videos_dir}")
                    print("\nPlease add a traffic video to that folder first.")
                    success = False
            else:
                print(f" Directory not found: {test_videos_dir}")
                success = False
        
        elif choice == "4":
            # Quick calibration test
            print("\n Quick Calibration Test")
            print("-" * 30)
            
            # Use default calibration values
            known_distance = 3.5  # Standard lane width in meters
            pixel_distance = 175  # Approximate pixels for lane width in typical video
            
            detector.set_calibration(known_distance, pixel_distance, detector.speed_limit)
            
            # Try to find a test video
            test_videos = [
                "data/test_videos/traffic.mp4",
                "data/test_videos/highway.mp4",
                "outputs/sample_traffic.mp4"
            ]
            
            for test_video in test_videos:
                if os.path.exists(test_video):
                    print(f"\n Testing with: {test_video}")
                    success = detector.process_video(
                        video_path=test_video,
                        output_path=f"outputs/calibration_test.mp4",
                        show_display=True,
                        save_output=True
                    )
                    break
            else:
                print(" No test videos found. Please add a video to data/test_videos/")
                success = False
        
        else:
            print(" Invalid choice")
=======
                        print("❌ Invalid selection")
                        success = False
                else:
                    print(f"❌ No videos found in {test_videos_dir}")
                    print("\nPlease add a traffic video to that folder first.")
                    success = False
            else:
                print(f"❌ Directory not found: {test_videos_dir}")
                success = False
        
        elif choice == "4":
            # Quick test on single image
            val_dir = "data/ua_detrac/valid/images"
            if os.path.exists(val_dir):
                import glob
                images = glob.glob(os.path.join(val_dir, "*.jpg"))
                if images:
                    test_image = images[0]
                    print(f"\n📷 Testing on image: {os.path.basename(test_image)}")
                    
                    results = detector.model(test_image, save=True, save_dir="outputs/test_images/", device='cpu')
                    
                    print(f"✅ Result saved to: outputs/test_images/")
                    success = True
                else:
                    print("❌ No test images found")
                    success = False
            else:
                print(f"❌ Validation directory not found: {val_dir}")
                success = False
        
        else:
            print("❌ Invalid choice")
>>>>>>> c1c1127 (Initial commit)
            success = False
    
    # Print statistics
    if success:
        detector.print_statistics()
<<<<<<< HEAD
        print("\n Processing complete!")
        
        # Show output files
        if os.path.exists("outputs"):
            print(f"\n Output files in 'outputs/' directory:")
            speed_files = []
            other_files = []
            
            for root, dirs, files in os.walk("outputs"):
                for file in files:
                    if 'speed' in file.lower():
                        speed_files.append(os.path.join(root, file))
                    elif file.endswith(('.mp4', '.jpg', '.png', '.csv')):
                        other_files.append(os.path.join(root, file))
            
            if speed_files:
                print("  SPEED-RELATED FILES:")
                for filepath in speed_files[:5]:
                    size = os.path.getsize(filepath) // 1024
                    filename = os.path.basename(filepath)
                    print(f"    - {filename} ({size} KB)")
            
            if other_files:
                print("\n  OTHER OUTPUT FILES:")
                for filepath in other_files[:5]:
                    size = os.path.getsize(filepath) // 1024
                    filename = os.path.basename(filepath)
                    print(f"    - {filename} ({size} KB)")
            
            # Save detailed speed report
            if hasattr(detector, 'speeds_history') and detector.speeds_history:
                report_file = "outputs/speed_analysis_report.txt"
                with open(report_file, 'w') as f:
                    f.write("SPEED ANALYSIS REPORT\n")
                    f.write("=" * 50 + "\n")
                    f.write(f"Speed Limit: {detector.speed_limit} km/h\n")
                    f.write(f"Violations Detected: {detector.stats['speed_violations']}\n")
                    f.write(f"Maximum Speed: {detector.stats['max_speed']:.1f} km/h\n")
                    f.write(f"Average Speed: {detector.stats['average_speed']:.1f} km/h\n")
                    f.write("\nCalibration Info:\n")
                    if hasattr(detector.speed_detector, 'calibration'):
                        for key, value in detector.speed_detector.calibration.items():
                            f.write(f"  {key}: {value}\n")
                
                print(f"\n Detailed speed report saved: {report_file}")
    else:
        print("\n Processing failed")
=======
        print("\n✅ Processing complete!")
        
        # Show output files
        if os.path.exists("outputs"):
            print(f"\n📁 Output files in 'outputs/' directory:")
            for root, dirs, files in os.walk("outputs"):
                for file in files[:10]:
                    if file.endswith(('.mp4', '.jpg', '.png')):
                        filepath = os.path.join(root, file)
                        size = os.path.getsize(filepath) // 1024
                        print(f"  - {file} ({size} KB)")
    else:
        print("\n❌ Processing failed")
>>>>>>> c1c1127 (Initial commit)

if __name__ == "__main__":
    main()