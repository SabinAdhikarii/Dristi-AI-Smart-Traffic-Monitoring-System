from ultralytics import YOLO
import cv2
import os
import sys
import time
import argparse
import numpy as np

# Import the speed detection modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from speed_detector import SpeedDetector
from vehicle_tracker import SimpleTracker


class TrafficVideoDetector:
    def __init__(self, model_path="models/yolov8n_quick_test/weights/best.pt"):
        print("Initializing Traffic Video Detector...")

        if not os.path.exists(model_path):
            print(f"❌ ERROR: Model not found at {model_path}")
            possible_paths = [
                "models/yolov8n_quick_test/weights/best.pt",
                "runs/detect/train/weights/best.pt",
                "runs/detect/train2/weights/best.pt",
                "runs/detect/train3/weights/best.pt",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    print(f"✅ Found alternative model: {model_path}")
                    break
            else:
                print("❌ No trained models found!")
                sys.exit(1)

        print(f"📦 Loading model: {model_path}")
        try:
            self.model = YOLO(model_path)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            sys.exit(1)

        self.class_names = self.model.names if hasattr(self.model, 'names') else {}
        self.tracker = SimpleTracker()
        self.speed_detector = SpeedDetector()

        self.class_colors = {
            'car': (0, 255, 0),
            'bus': (255, 0, 0),
            'truck': (0, 0, 255),
            'van': (255, 255, 0),
        }

        self.speed_limit = 50
        self.calibration_set = False
        self.stats = {}
        self.speeds_history = []
        self.max_width = 1280
        self.max_height = 720

    # keep all your methods here: set_calibration, _resize_to_fit_screen, process_video, _add_overlay, print_statistics


def create_sample_video():
    """Create a sample video from validation images"""
    import glob
    val_dir = "data/ua_detrac/valid/images"
    if not os.path.exists(val_dir):
        print(f"❌ Validation directory not found: {val_dir}")
        return None

    images = sorted(glob.glob(os.path.join(val_dir, "*.jpg")))[:20]
    if not images:
        print("❌ No validation images found")
        return None

    img = cv2.imread(images[0])
    if img is None:
        print("❌ Could not read image")
        return None

    height, width = img.shape[:2]
    sample_video = "outputs/sample_traffic.mp4"
    os.makedirs("outputs", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(sample_video, fourcc, 5.0, (width, height))

    print(f"📹 Creating sample video from {len(images)} images...")
    for img_path in images:
        img = cv2.imread(img_path)
        if img is not None:
            out.write(img)

    out.release()
    print(f"✅ Created sample video: {sample_video}")
    return sample_video


def main():
    parser = argparse.ArgumentParser(description="Traffic Video Vehicle Detection with SPEED MEASUREMENT")
    parser.add_argument("--video", type=str, help="Path to input video file")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    parser.add_argument("--model", type=str, default="models/yolov8n_quick_test/weights/best.pt")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--max-height", type=int, default=720)
    parser.add_argument("--speed-limit", type=int, default=50)
    parser.add_argument("--calibrate", action="store_true")

    args = parser.parse_args()
    os.makedirs("outputs", exist_ok=True)

    detector = TrafficVideoDetector(args.model)
    detector.speed_limit = args.speed_limit
    detector.max_width = args.max_width
    detector.max_height = args.max_height

    # Calibration option
    if args.calibrate:
        try:
            known_dist = float(input("Enter known distance in meters (e.g., 3.5): "))
            pixel_dist = float(input("Enter that distance in pixels: "))
            detector.set_calibration(known_dist, pixel_dist, detector.speed_limit)
        except ValueError:
            print("Invalid input. Using default calibration.")

    # Upload mode
    if args.upload:
        video_path = input("Enter path to your traffic video: ").strip()
        if not os.path.exists(video_path):
            print("❌ Invalid path")
            return
        detector.process_video(video_path, args.output, not args.no_display, not args.no_save)

    # Direct video argument
    elif args.video:
        detector.process_video(args.video, args.output, not args.no_display, not args.no_save)

    # Interactive menu
    else:
        print("\nChoose an option:")
        print("1. Create and test with sample video")
        print("2. Upload your own traffic video")
        print("3. Test on video in data/test_videos/")
        print("4. Quick test with calibration")

        choice = input("Enter choice (1/2/3/4): ").strip()

        if choice == "1":
            sample_video = create_sample_video()
            if sample_video:
                detector.process_video(sample_video, None, True, True)

        elif choice == "2":
            video_path = input("Enter path to your traffic video: ").strip()
            if os.path.exists(video_path):
                detector.process_video(video_path, None, True, True)
            else:
                print("❌ Invalid path")

        elif choice == "3":
            test_video = "data/test_videos/test.mp4"
            if os.path.exists(test_video):
                detector.process_video(test_video, None, True, True)
            else:
                print("❌ Test video not found")

        elif choice == "4":
            try:
                known_dist = float(input("Enter known distance in meters (e.g., 3.5): "))
                pixel_dist = float(input("Enter that distance in pixels: "))
                detector.set_calibration(known_dist, pixel_dist, detector.speed_limit)
                video_path = input("Enter path to your traffic video: ").strip()
                if os.path.exists(video_path):
                    detector.process_video(video_path, None, True, True)
                else:
                    print("❌ Invalid path")
            except ValueError:
                print("Invalid input.")


if __name__ == "__main__":
    main()