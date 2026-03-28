#!/usr/bin/env python3
"""
Test the trained model on your videos - Windows optimized with project paths
"""
from ultralytics import YOLO
import os
import torch
from pathlib import Path

# Get project root directory
PROJECT_ROOT = os.getcwd()
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TEST_VIDEOS_DIR = os.path.join(DATA_DIR, "test_videos")
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")

def find_latest_model():
    """Find the latest trained model in models folder"""
    model_paths = []
    
    # Search for all .pt files in models directory
    if os.path.exists(MODELS_DIR):
        for root, dirs, files in os.walk(MODELS_DIR):
            for file in files:
                if file.endswith('.pt') and ('best' in file or 'last' in file):
                    full_path = os.path.join(root, file)
                    model_paths.append(full_path)
    
    # Sort by modification time (newest first)
    model_paths.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    return model_paths[0] if model_paths else None

def check_gpu():
    """Check if GPU is available"""
    if torch.cuda.is_available():
        device = 0
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU detected: {gpu_name}")
        return device
    else:
        print("No GPU detected. Using CPU (slower)")
        return 'cpu'

def find_test_videos():
    """Find all test videos in test_videos folder"""
    videos = []
    if os.path.exists(TEST_VIDEOS_DIR):
        for file in os.listdir(TEST_VIDEOS_DIR):
            if file.endswith(('.mp4', '.avi', '.mov', '.mkv', '.MP4')):
                videos.append(os.path.join(TEST_VIDEOS_DIR, file))
    return videos

def test_model():
    """Main test function"""
    print("=" * 60)
    print("TRAFFIC VIDEO DETECTION TEST")
    print("=" * 60)
    print(f"Project: {PROJECT_ROOT}")
    
    # Create necessary directories
    os.makedirs(RUNS_DIR, exist_ok=True)
    
    # Find model
    print("\nLooking for trained model...")
    model_path = find_latest_model()
    
    if not model_path:
        print("No trained model found in models folder!")
        print("Please train a model first using: python src/train.py")
        return
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Show model info
    print(f"Model classes: {list(model.names.values())}")
    
    # Check GPU
    device = check_gpu()
    
    # Find test videos
    print("\nLooking for test videos...")
    test_videos = find_test_videos()
    
    if not test_videos:
        print(f"No videos found in {TEST_VIDEOS_DIR}")
        print("Please add test videos to the test_videos folder")
        return
    
    print(f"Found {len(test_videos)} test video(s):")
    for i, video in enumerate(test_videos, 1):
        print(f"  {i}. {os.path.basename(video)}")
    
    # Process each video
    for video_path in test_videos:
        print(f"\nProcessing: {os.path.basename(video_path)}")
        print(f"Full path: {video_path}")
        
        # Create output name based on video filename
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_name = f"result_{video_name}"
        
        try:
            # Run detection with stream=True for memory efficiency
            results = model(
                source=video_path,
                conf=0.25,                 # Confidence threshold
                iou=0.45,                   # IoU threshold for NMS
                save=True,                   # Save the output video
                device=device,                # Use GPU if available
                show=True,                    # Show the video while processing
                project=RUNS_DIR,             # Save to project runs folder
                name=output_name,              # Output folder name
                exist_ok=True,                  # Overwrite if exists
                verbose=False,                   # Less console output
                stream=True                      # Use streaming mode for videos
            )
            
            # Process results generator
            for result in results:
                # You can access detection information here if needed
                # boxes = result.boxes  # Bounding boxes
                # names = result.names  # Class names
                pass
            
            # Show results location
            output_path = os.path.join(RUNS_DIR, output_name)
            print(f"Completed: {os.path.basename(video_path)}")
            print(f"Results saved to: {output_path}")
            
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)
    print("\nResults saved in:")
    print(f"  {RUNS_DIR}")
    print("\nEach video result is in its own folder with:")
    print("  - Annotated video")
    print("  - Detection labels")
    print("  - Frames with detections")
    
def quick_test():
    """Quick test with default parameters"""
    print("\n" + "=" * 60)
    print("QUICK TEST MODE")
    print("=" * 60)
    
    # Find model
    model_path = find_latest_model()
    if not model_path:
        print("No model found")
        return
    
    print(f"Using model: {model_path}")
    model = YOLO(model_path)
    
    # Find first test video
    videos = find_test_videos()
    if not videos:
        print("No test videos found")
        return
    
    video_path = videos[0]
    print(f"Testing on: {os.path.basename(video_path)}")
    
    # Run quick test with streaming
    device = check_gpu()
    results = model(
        source=video_path,
        conf=0.25,
        save=True,
        device=device,
        show=True,
        project=RUNS_DIR,
        name="quick_test",
        exist_ok=True,
        max_det=50,
        stream=True
    )
    
    # Process results generator
    for result in results:
        pass
    
    print(f"\nQuick test complete")
    print(f"Results: {os.path.join(RUNS_DIR, 'quick_test')}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test trained model on videos")
    parser.add_argument("--quick", action="store_true", help="Run quick test on first video")
    parser.add_argument("--video", type=str, help="Test on specific video file")
    parser.add_argument("--model", type=str, help="Specific model path to use")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    
    args = parser.parse_args()
    
    if args.quick:
        quick_test()
    elif args.video:
        # Test single video with custom model
        model_path = args.model if args.model else find_latest_model()
        if not model_path:
            print("No model found")
            exit()
        
        print(f"Loading model: {model_path}")
        model = YOLO(model_path)
        
        video_path = args.video
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            exit()
        
        device = check_gpu()
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        results = model(
            source=video_path,
            conf=args.conf,
            save=True,
            device=device,
            show=True,
            project=RUNS_DIR,
            name=f"result_{video_name}",
            exist_ok=True,
            stream=True
        )
        
        # Process results generator
        for result in results:
            pass
        
        print(f"\nResults saved to: {os.path.join(RUNS_DIR, f'result_{video_name}')}")
    else:
        test_model()