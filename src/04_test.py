#!/usr/bin/env python3
"""
Test trained YOLOv8 model on videos
"""
from ultralytics import YOLO
import cv2
import argparse
import os
from pathlib import Path

def test_on_image(model_path, image_path):
    """Test model on a single image"""
    print(f"\nTesting on image: {image_path}")
    
    # Load model
    model = YOLO(model_path)
    
    # Run inference
    results = model(image_path, save=True, save_dir="outputs/images/")
    
    # Show results
    for r in results:
        im_array = r.plot()  # Plot results
        cv2.imshow("Detection", im_array)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def test_on_video(model_path, video_path):
    """Test model on video"""
    print(f"\nProcessing video: {video_path}")
    
    # Load model
    model = YOLO(model_path)
    
    # Process video
    results = model(
        video_path,
        stream=True,      # Stream mode for videos
        save=True,        # Save output video
        save_dir="outputs/videos/",
        show=True,        # Show in real-time
        conf=0.25,        # Confidence threshold
        iou=0.45,         # IoU threshold
        device="0"        # Use GPU 0
    )
    
    print("\n✅ Processing complete!")
    print(f"Output saved to: outputs/videos/")

def test_on_validation_set(model_path):
    """Test model on validation set"""
    print("\nTesting on validation set...")
    
    model = YOLO(model_path)
    
    # Run validation
    metrics = model.val(
        data="data/ua_detrac/data.yaml",
        split="val",
        device="0"
    )
    
    print(f"\nValidation Results:")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  mAP50:    {metrics.box.map50:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Test YOLOv8 model")
    parser.add_argument("--model", type=str, default="models/yolov8n_ua_detrac/weights/best.pt",
                       help="Path to trained model")
    parser.add_argument("--video", type=str, help="Path to test video")
    parser.add_argument("--image", type=str, help="Path to test image")
    parser.add_argument("--validate", action="store_true", help="Test on validation set")
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs("outputs/videos", exist_ok=True)
    os.makedirs("outputs/images", exist_ok=True)
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        print("\nAvailable models:")
        for root, dirs, files in os.walk("models"):
            for file in files:
                if file.endswith(".pt"):
                    print(f"  {os.path.join(root, file)}")
        return
    
    print(f"✅ Loading model: {args.model}")
    
    # Run tests
    if args.validate:
        test_on_validation_set(args.model)
    elif args.video:
        test_on_video(args.model, args.video)
    elif args.image:
        test_on_image(args.model, args.image)
    else:
        # Test on sample validation image
        sample_image = "data/ua_detrac/valid/images/"
        images = os.listdir(sample_image)
        if images:
            test_on_image(args.model, os.path.join(sample_image, images[0]))
        else:
            print("No test images found. Please provide --video or --image")

if __name__ == "__main__":
    main()