import numpy as np
from collections import defaultdict, deque
import json
import os

class SpeedDetector:
    def __init__(self, calibration_file=None):
        """
        Initialize speed detector with camera calibration.
        calibration_file: JSON with 'pixels_per_meter' or use default
        """
        self.track_history = defaultdict(lambda: deque(maxlen=30))  # Store last 30 positions per vehicle
        self.speeds = {}  # Current speeds for each vehicle ID
        
        # Camera calibration (pixels to meters conversion)
        if calibration_file and os.path.exists(calibration_file):
            with open(calibration_file, 'r') as f:
                self.calibration = json.load(f)
        else:
            # Default calibration (approx for Nepali roads)
            # You'll need to calibrate for each camera location!
            self.calibration = {
                'pixels_per_meter': 50,  # Default: 50 pixels = 1 meter
                'fps': 30,  # Default frames per second
                'speed_limit': 50  # Default speed limit in km/h
            }
    
    def calibrate_camera(self, known_distance_meters, pixel_distance):
        """
        Manually calibrate camera by measuring known distance in video.
        known_distance_meters: Real-world distance (e.g., lane width = 3.5m)
        pixel_distance: Pixel length of that distance in video frame
        """
        self.calibration['pixels_per_meter'] = pixel_distance / known_distance_meters
        print(f"Calibrated: {self.calibration['pixels_per_meter']:.2f} pixels/meter")
        return self.calibration
    
    def update_tracks(self, detections, frame_count):
        """
        Update vehicle tracks with new detections.
        detections: List of [x1, y1, x2, y2, confidence, class_id, track_id]
        Returns: Updated speeds dictionary
        """
        for det in detections:
            if len(det) < 7:  # Ensure we have track_id
                continue
                
            x1, y1, x2, y2, conf, class_id, track_id = det
            centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            # Add to track history
            self.track_history[track_id].append((frame_count, centroid, class_id))
        
        return self.get_all_speeds(self.calibration.get('fps', 30))
    
    def calculate_speed(self, track_id, fps=30):
        """
        Calculate speed for a specific vehicle track.
        Returns speed in km/h
        """
        if track_id not in self.track_history or len(self.track_history[track_id]) < 2:
            return 0.0
        
        track = list(self.track_history[track_id])
        
        # Use last 1 second of data for stability
        current_frame = track[-1][0] if track else 0
        recent_frames = [t for t in track if t[0] > current_frame - fps]
        
        if len(recent_frames) < 2:
            return 0.0
        
        # Calculate total pixel movement
        total_pixels = 0
        for i in range(1, len(recent_frames)):
            _, (x1, y1), _ = recent_frames[i-1]
            _, (x2, y2), _ = recent_frames[i]
            total_pixels += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Convert to real-world speed
        avg_pixels_per_frame = total_pixels / (len(recent_frames) - 1)
        pixels_per_second = avg_pixels_per_frame * fps
        meters_per_second = pixels_per_second / self.calibration['pixels_per_meter']
        km_per_hour = meters_per_second * 3.6
        
        # Store calculated speed
        self.speeds[track_id] = km_per_hour
        
        return km_per_hour
    
    def get_all_speeds(self, fps=30):
        """Calculate speeds for all tracked vehicles."""
        speeds = {}
        for track_id in list(self.track_history.keys()):
            speed = self.calculate_speed(track_id, fps)
            if speed > 0.5:  # Ignore very slow/stationary vehicles
                speeds[track_id] = speed
        return speeds
    
    def detect_speed_violations(self, speed_limit=None):
        """Identify vehicles exceeding speed limit."""
        if speed_limit is None:
            speed_limit = self.calibration.get('speed_limit', 50)
        
        violations = []
        for track_id, speed in self.speeds.items():
            if speed > speed_limit:
                # Get vehicle info from track history
                if track_id in self.track_history and self.track_history[track_id]:
                    track_list = list(self.track_history[track_id])
                    if track_list:
                        _, centroid, class_id = track_list[-1]
                        violations.append({
                            'track_id': track_id,
                            'speed': speed,
                            'speed_limit': speed_limit,
                            'class_id': class_id,
                            'position': centroid
                        })
        return violations
    
    def get_calibration_info(self):
        """Get current calibration settings."""
        return self.calibration.copy()
    
    def reset(self):
        """Reset all tracks and speeds."""
        self.track_history.clear()
        self.speeds.clear()