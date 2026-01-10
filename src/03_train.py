#!/usr/bin/env python3
"""
Train YOLOv8 on UA-DETRAC dataset with automatic path fixing
"""
from ultralytics import YOLO
import os
import yaml
from pathlib import Path

def fix_data_yaml():
    """Fix the paths in data.yaml if needed"""
    yaml_path = "data/ua_detrac/data.yaml"
    
    if not os.path.exists(yaml_path):
        print(f" {yaml_path} not found!")
        return False
    
    with open(yaml_path, 'r') as f:
        content = f.read()
    
    # Check if paths are wrong
    if "../train/images" in content:
        print("  Detected wrong paths in data.yaml. Fixing...")
        
        # Fix the paths
        content = content.replace("../train/images", "train/images")
        content = content.replace("../valid/images", "valid/images")
        content = content.replace("../test/images", "test/images")
        
        # Save fixed version
        with open(yaml_path, 'w') as f:
            f.write(content)
        
        print(" Fixed data.yaml paths")
        return True
    
    return True

def check_dataset_structure():
    """Verify dataset structure is correct"""
    print("\n" + "=" * 60)
    print("Dataset Structure Check")
    print("=" * 60)
    
    checks = []
    
    # Check 1: Directories exist
    dirs_to_check = [
        ("data/ua_detrac/train/images", True),
        ("data/ua_detrac/train/labels", True),
        ("data/ua_detrac/valid/images", True),
        ("data/ua_detrac/valid/labels", True),
    ]
    
    for path, required in dirs_to_check:
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        checks.append((path, exists, required))
        print(f"{status} {path}")
    
    # Check 2: Count files
    train_images = len(os.listdir("data/ua_detrac/train/images"))
    train_labels = len(os.listdir("data/ua_detrac/train/labels"))
    print(f"\n Training set: {train_images} images, {train_labels} labels")
    
    val_images = len(os.listdir("data/ua_detrac/valid/images"))
    val_labels = len(os.listdir("data/ua_detrac/valid/labels"))
    print(f" Validation set: {val_images} images, {val_labels} labels")
    
    # Check 3: Verify label format
    print("\n Checking label format...")
    try:
        sample_label = os.listdir("data/ua_detrac/train/labels")[0]
        with open(f"data/ua_detrac/train/labels/{sample_label}", 'r') as f:
            first_line = f.readline().strip()
            print(f"Sample label format: {first_line}")
            if len(first_line.split()) == 5:
                print(" YOLO format correct")
            else:
                print("  Unexpected label format")
    except:
        print("  Could not check label format")
    
    # Check 4: Verify data.yaml
    print("\n Checking data.yaml...")
    try:
        with open("data/ua_detrac/data.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"  Classes: {config.get('nc')}")
        print(f"  Class names: {config.get('names')}")
        print(f"  Train path: {config.get('train')}")
        print(f"  Val path: {config.get('val')}")
        
        # Verify paths exist
        train_path = os.path.join("data/ua_detrac", config.get('train', ''))
        if os.path.exists(train_path):
            print(f" Train path exists: {train_path}")
        else:
            print(f" Train path not found: {train_path}")
            
    except Exception as e:
        print(f" Error reading data.yaml: {e}")
    
    print("\n" + "=" * 60)
    
    # Return True if all required checks pass
    all_good = all([exists for path, exists, required in checks if required])
    return all_good

def train_model():
    """Train YOLOv8 model"""
    print("\n" + "=" * 60)
    print("Starting YOLOv8 Training")
    print("=" * 60)
    
    # 1. Fix paths if needed
    if not fix_data_yaml():
        print(" Failed to fix data.yaml")
        return None, None
    
    # 2. Check dataset
    if not check_dataset_structure():
        print(" Dataset structure issues detected")
        response = input("Continue anyway? (y/n): ").lower()
        if response != 'y':
            print("Training cancelled.")
            return None, None
    
    # 3. Load model
    print("\n[1] Loading YOLOv8n model...")
    try:
        model = YOLO("yolov8n.pt")  # Pre-trained on COCO
    except Exception as e:
        print(f" Failed to load model: {e}")
        print("Trying alternative...")
        model = YOLO("yolov8n.yaml")  # Create from scratch
    
    # 4. Train
    print("\n[2] Starting training...")
    print("   This may take 30-60 minutes")
    print("   Press Ctrl+C to stop early\n")
    
    try:
        results = model.train(
            data="data/ua_detrac/data.yaml",
            epochs=50,
            imgsz=640,
            batch=16,
            patience=10,
            device="0",  # Change to "cpu" if no GPU
            project="models",
            name="yolov8n_ua_detrac",
            exist_ok=True,
            verbose=True,
            workers=4,  # Reduce if you get errors
        )
        print("\n Training completed successfully!")
        return model, results
        
    except KeyboardInterrupt:
        print("\n  Training interrupted by user")
        return model, None
    except Exception as e:
        print(f"\n Training failed: {e}")
        return None, None

def quick_test_model(model_path):
    """Quick test to verify model works"""
    print("\n" + "=" * 60)
    print("Quick Model Test")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        print(f" Model not found: {model_path}")
        return
    
    model = YOLO(model_path)
    
    # Test on a single validation image
    val_images = os.listdir("data/ua_detrac/valid/images")
    if val_images:
        test_image = os.path.join("data/ua_detrac/valid/images", val_images[0])
        print(f"Testing on: {test_image}")
        
        results = model(test_image, save=True, save_dir="outputs/quick_test/")
        print(" Quick test complete!")
        print(f"Results saved to: outputs/quick_test/")
    else:
        print("  No validation images found for testing")

def main():
    """Main training pipeline"""
    
    print(" UA-DETRAC Vehicle Detection Training")
    print("=" * 60)
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    # Ask for training parameters
    print("\nTraining Configuration:")
    print("  Model: YOLOv8n (fastest)")
    print("  Epochs: 50")
    print("  Image size: 640x640")
    print("  Batch size: 16")
    
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except:
        pass
    
    if gpu_available:
        print(f"  Device: GPU ({torch.cuda.get_device_name(0)})")
    else:
        print("  Device: CPU (slow)")
        print("    Training will be slow on CPU!")
    
    confirm = input("\nStart training? (y/n): ").lower()
    if confirm != 'y':
        print("Training cancelled.")
        return
    
    # Run training
    model, results = train_model()
    
    if model is not None:
        # Quick test
        model_path = "models/yolov8n_ua_detrac/weights/best.pt"
        if os.path.exists(model_path):
            quick_test_model(model_path)
            
            print("\n" + "=" * 60)
            print(" TRAINING COMPLETE!")
            print("=" * 60)
            print(f"\nTrained model saved to: {model_path}")
            print("\nNext steps:")
            print("1. Test on videos: python src/04_test.py --validate")
            print("2. Test on your video: python src/04_test.py --video your_video.mp4")
            print("3. View training results: open models/yolov8n_ua_detrac/")
        else:
            print("\n  Model not found. Check training logs.")
    
    print("\nDone!")

if __name__ == "__main__":
    main()