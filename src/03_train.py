#!/usr/bin/env python3
"""
Train YOLOv8 on UA-DETRAC dataset - Windows optimized with project paths
"""
from ultralytics import YOLO
import os
import yaml
import torch

# Get project root directory
PROJECT_ROOT = os.getcwd()
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "ua_detrac")
DATA_YAML = os.path.join(DATA_DIR, "data.yaml")

def fix_data_yaml():
    """Fix the paths in data.yaml if needed"""
    if not os.path.exists(DATA_YAML):
        print(f" {DATA_YAML} not found!")
        return False
    
    with open(DATA_YAML, 'r') as f:
        content = f.read()
    
    # Fix paths for Windows
    if "../train/images" in content or "train/images" not in content:
        print("  Fixing paths in data.yaml...")
        
        # Ensure correct relative paths
        new_content = f"""# Dataset configuration
path: {PROJECT_ROOT}  # Project root
train: data/ua_detrac/train/images
val: data/ua_detrac/valid/images
test: data/ua_detrac/test/images

nc: 4
names: ['bus', 'car', 'truck', 'van']
"""
        
        with open(DATA_YAML, 'w') as f:
            f.write(new_content)
        
        print(" Fixed data.yaml paths")
        return True
    
    return True

def check_dataset_structure():
    """Verify dataset structure is correct"""
    print("\n" + "=" * 60)
    print(" Dataset Structure Check")
    print("=" * 60)
    
    checks_passed = True
    
    # Check directories
    dirs_to_check = [
        os.path.join(DATA_DIR, "train", "images"),
        os.path.join(DATA_DIR, "train", "labels"),
        os.path.join(DATA_DIR, "valid", "images"),
        os.path.join(DATA_DIR, "valid", "labels"),
    ]
    
    for path in dirs_to_check:
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"{status} {os.path.relpath(path, PROJECT_ROOT)}")
        if not exists:
            checks_passed = False
    
    # Count files
    try:
        train_images = len(os.listdir(os.path.join(DATA_DIR, "train", "images")))
        train_labels = len(os.listdir(os.path.join(DATA_DIR, "train", "labels")))
        print(f"\n Training set: {train_images} images, {train_labels} labels")
        
        val_images = len(os.listdir(os.path.join(DATA_DIR, "valid", "images")))
        val_labels = len(os.listdir(os.path.join(DATA_DIR, "valid", "labels")))
        print(f" Validation set: {val_images} images, {val_labels} labels")
    except Exception as e:
        print(f" Error counting files: {e}")
        checks_passed = False
    
    # Check label format
    print("\n Checking label format...")
    try:
        sample_label = os.listdir(os.path.join(DATA_DIR, "train", "labels"))[0]
        label_path = os.path.join(DATA_DIR, "train", "labels", sample_label)
        with open(label_path, 'r') as f:
            first_line = f.readline().strip()
            parts = first_line.split()
            if len(parts) == 5:
                print(f" YOLO format correct: {first_line[:50]}...")
            else:
                print(f" Unexpected format: {first_line}")
                checks_passed = False
    except Exception as e:
        print(f" Could not check label format: {e}")
        checks_passed = False
    
    return checks_passed

def train_model():
    """Train YOLOv8 model"""
    print("\n" + "=" * 60)
    print(" Starting YOLOv8 Training")
    print("=" * 60)
    
    # Fix paths
    if not fix_data_yaml():
        print(" Failed to fix data.yaml")
        return None, None
    
    # Check dataset
    if not check_dataset_structure():
        print("\n  Dataset structure has issues")
        response = input("Continue anyway? (y/n): ").lower()
        if response != 'y':
            print("Training cancelled.")
            return None, None
    
    # Check GPU
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        device = 0
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n GPU detected: {gpu_name}")
    else:
        device = 'cpu'
        print("\n No GPU detected. Training on CPU (will be slower)")
    
    # Load model
    print("\n Loading YOLOv8n model...")
    try:
        model = YOLO("yolov8n.pt")
        print(" Model loaded successfully")
    except Exception as e:
        print(f" Failed to load model: {e}")
        return None, None
    
    # Training configuration
    print("\nTraining Configuration:")
    print(f"   Data: {os.path.relpath(DATA_YAML, PROJECT_ROOT)}")
    print(f"   Epochs: 10")
    print(f"   Image size: 640")
    print(f"   Batch size: 32")
    print(f"   Device: {device}")
    print(f"   Project: {os.path.relpath(MODELS_DIR, PROJECT_ROOT)}")
    
    # Train
    print("\n  Starting training...")
    print("   (Press Ctrl+C to stop early)\n")
    
    try:
        results = model.train(
            data=DATA_YAML,
            epochs=10,
            imgsz=640,
            batch=32,
            device=device,
            project=MODELS_DIR,           # Save to project/models/
            name="yolov8n_trained",       # models/yolov8n_trained/
            exist_ok=True,
            verbose=True,
            workers=4,
            amp=True,                      # Mixed precision
            cache=False,                    # Don't cache images
            patience=10,                     # Early stopping
            save=True,
            save_period=5
        )
        
        print("\n" + "=" * 60)
        print(" TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        return model, results
        
    except KeyboardInterrupt:
        print("\n  Training interrupted by user")
        return model, None
    except Exception as e:
        print(f"\n Training failed: {e}")
        return None, None

def show_results():
    """Show where models are saved"""
    print("\n" + "=" * 60)
    print(" Saved Files Location")
    print("=" * 60)
    
    # Check for trained model
    model_path = os.path.join(MODELS_DIR, "yolov8n_trained", "weights", "best.pt")
    if os.path.exists(model_path):
        print(f"\n Trained model: {os.path.relpath(model_path, PROJECT_ROOT)}")
        print(f"   File size: {os.path.getsize(model_path) / 1024 / 1024:.1f} MB")
        
        # Load and show classes
        model = YOLO(model_path)
        print(f"   Classes: {list(model.names.values())}")
    else:
        print("\n No trained model found")
    
    # Show all model files
    print("\n All model files:")
    if os.path.exists(MODELS_DIR):
        for root, dirs, files in os.walk(MODELS_DIR):
            for file in files:
                if file.endswith('.pt'):
                    full_path = os.path.join(root, file)
                    print(f"   - {os.path.relpath(full_path, PROJECT_ROOT)}")

def main():
    """Main training pipeline"""
    
    print("=" * 60)
    print(" UA-DETRAC VEHICLE DETECTION TRAINING")
    print("=" * 60)
    print(f" Project: {PROJECT_ROOT}")
    
    # Create necessary directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "outputs"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "runs"), exist_ok=True)
    
    # Check if dataset exists
    if not os.path.exists(DATA_DIR):
        print(f"\n Dataset not found at: {DATA_DIR}")
        print("Please ensure your UA-DETRAC dataset is in:")
        print(f"   {DATA_DIR}")
        return
    
    # Confirm training
    print("\n Training Summary:")
    print(f"   Model: YOLOv8n")
    print(f"   Epochs: 10")
    print(f"   Dataset: UA-DETRAC (4 classes)")
    print(f"   Save location: {os.path.relpath(MODELS_DIR, PROJECT_ROOT)}")
    
    confirm = input("\nStart training? (y/n): ").lower()
    if confirm != 'y':
        print("Training cancelled.")
        return
    
    # Run training
    model, results = train_model()
    
    # Show results
    show_results()
    
    print("\n" + "=" * 60)
    print(" SETUP COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Test your model on videos:")
    print("   python src/test_with_model.py")
    print("\n2. Check your model at:")
    print(f"   {os.path.join(MODELS_DIR, 'yolov8n_trained', 'weights', 'best.pt')}")
    print("\n3. All outputs will be saved in:")
    print("   - models/     : Trained model weights")
    print("   - runs/       : Detection results")
    print("   - outputs/    : Additional outputs")

if __name__ == "__main__":
    main()