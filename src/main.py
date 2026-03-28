"""
Main Traffic Monitoring System
Runs all detections when user uploads a video
"""

import os
import json
import cv2
from datetime import datetime
from red_light_violation import RedLightViolationDetector
from speed_detector import SpeedDetector
from helmet_detection import HelmetDetector

class TrafficMonitoringSystem:
    def __init__(self):
        print("=" * 60)
        print("TRAFFIC MONITORING SYSTEM - MAIN CONTROLLER")
        print("=" * 60)
        
        # Model paths
        self.vehicle_model = "models/yolov8n_trained/weights/best.pt"
        self.traffic_light_model = "models/traffic_light_detector/weights/best.pt"
        self.helmet_model = "models/helmet_detector_kaggle/weights/best.pt"
        
        # Create output directory for this session
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join("results", self.session_id)
        os.makedirs(self.session_dir, exist_ok=True)
        
        print(f"Session ID: {self.session_id}")
        print(f"Results will be saved in: {self.session_dir}")
    
    def process_video(self, video_path):
        """Process a single video with all detectors"""
        print(f"\n{'='*60}")
        print(f"PROCESSING VIDEO: {video_path}")
        print(f"{'='*60}")
        
        # Store all results
        all_results = {
            'session_id': self.session_id,
            'video': video_path,
            'timestamp': datetime.now().isoformat(),
            'red_light': {},
            'speed': {},
            'helmet': {}
        }
        
        # 1. RED LIGHT VIOLATION DETECTION
        print("\n[1/3] Running Red Light Detection...")
        red_detector = RedLightViolationDetector(
            self.vehicle_model, 
            self.traffic_light_model
        )
        red_result = red_detector.process_video(
            video_path, 
            output_video_path=os.path.join(self.session_dir, "red_light_analyzed.mp4"),
            interactive=False  # Use auto-detection
        )
        all_results['red_light'] = red_result
        
        # 2. SPEED DETECTION
        print("\n[2/3] Running Speed Detection...")
        speed_detector = SpeedDetector(self.vehicle_model)
        speed_result = speed_detector.process_video(
            video_path,
            output_video_path=os.path.join(self.session_dir, "speed_analyzed.mp4"),
            calibrate=False
        )
        all_results['speed'] = speed_result
        
        # 3. HELMET DETECTION
        print("\n[3/3] Running Helmet Detection...")
        helmet_detector = HelmetDetector(self.helmet_model)
        helmet_result = helmet_detector.process_video(
            video_path,
            output_video_path=os.path.join(self.session_dir, "helmet_analyzed.mp4")
        )
        all_results['helmet'] = helmet_result
        
        # 4. COMBINE ALL RESULTS
        print("\n" + "="*60)
        print("GENERATING FINAL REPORT")
        print("="*60)
        
        final_report = self.generate_report(all_results)
        
        # 5. EXTRACT VIOLATION CLIPS
        print("\nExtracting violation clips...")
        self.extract_violation_clips(video_path, all_results)
        
        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"Results saved in: {self.session_dir}")
        print(f"Final report: {os.path.join(self.session_dir, 'final_report.json')}")
        
        return final_report
    
    def generate_report(self, all_results):
        """Combine all detection results into one report"""
        
        # Count violations
        red_count = len(all_results['red_light'].get('violations', []))
        speed_count = len(all_results['speed'].get('violations', []))
        helmet_count = len(all_results['helmet'].get('violations', []))
        
        report = {
            'session_id': self.session_id,
            'video_file': all_results['video'],
            'processing_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'summary': {
                'total_violations': red_count + speed_count + helmet_count,
                'red_light_violations': red_count,
                'speeding_violations': speed_count,
                'helmet_violations': helmet_count
            },
            'violations': []
        }
        
        # Add red light violations
        for v in all_results['red_light'].get('violations', []):
            v['type'] = 'red_light'
            report['violations'].append(v)
        
        # Add speed violations
        for v in all_results['speed'].get('violations', []):
            v['type'] = 'speeding'
            report['violations'].append(v)
        
        # Add helmet violations
        for v in all_results['helmet'].get('violations', []):
            v['type'] = 'helmet'
            report['violations'].append(v)
        
        # Sort by timestamp/frame
        report['violations'].sort(key=lambda x: x.get('frame', 0))
        
        # Save report
        report_path = os.path.join(self.session_dir, 'final_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def extract_violation_clips(self, video_path, all_results):
        """Extract short video clips for each violation"""
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        clips_dir = os.path.join(self.session_dir, "clips")
        os.makedirs(clips_dir, exist_ok=True)
        
        for violation in all_results.get('violations', []):
            frame_num = violation.get('frame', 0)
            if frame_num == 0:
                continue
            
            # Calculate clip time (5 seconds before and after)
            start_frame = max(0, frame_num - fps * 3)  # 3 seconds before
            end_frame = frame_num + fps * 2  # 2 seconds after
            
            # Extract clip
            clip_path = os.path.join(clips_dir, f"{violation['type']}_{violation['id']}.mp4")
            self.extract_clip(video_path, start_frame, end_frame, clip_path)
            violation['clip'] = clip_path
        
        cap.release()
    
    def extract_clip(self, video_path, start_frame, end_frame, output_path):
        """Extract a portion of video"""
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for frame_num in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        cap.release()
        out.release()

def main():
    # Create system
    system = TrafficMonitoringSystem()
    
    # Get video from user
    video_path = input("Enter path to traffic video: ").strip()
    
    if not os.path.exists(video_path):
        print(f"Error: Video not found: {video_path}")
        return
    
    # Process video
    results = system.process_video(video_path)
    
    # Show summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Total violations: {results['summary']['total_violations']}")
    print(f"  Red Light: {results['summary']['red_light_violations']}")
    print(f"  Speeding: {results['summary']['speeding_violations']}")
    print(f"  Helmet: {results['summary']['helmet_violations']}")
    
    print("\nResults saved in:")
    print(f"  {system.session_dir}/")

if __name__ == "__main__":
    main()