import os
import torch
from ultralytics import YOLO

def train_helmet_model():
    print("=" * 60)
    print("HELMET DETECTION TRAINING - KAGGLE DATASET")
    print("=" * 60)
    
    # Paths
    project_root = os.getcwd()
    data_yaml = os.path.join(project_root, "data", "helmet_kaggle", "data.yaml")
    models_dir = os.path.join(project_root, "models")
    
    # Check dataset
    if not os.path.exists(data_yaml):
        print(f"Error: Dataset not found at {data_yaml}")
        print("Please download the dataset from Kaggle and extract to data/helmet_kaggle/")
        return
    
    # Check dataset structure
    train_dir = os.path.join("data", "helmet_kaggle", "train", "images")
    if not os.path.exists(train_dir):
        print(f"Error: Training images not found at {train_dir}")
        return
    
    # Count images
    train_images = len(os.listdir(train_dir))
    print(f"Training images: {train_images}")
    
    # Check GPU
    if torch.cuda.is_available():
        device = 0
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU detected: {gpu_name}")
    else:
        device = 'cpu'
        print("No GPU detected. Training on CPU (slower)")
    
    # Load model
    print("\nLoading YOLOv8n model...")
    model = YOLO('yolov8n.pt')
    
    # Show dataset classes
    import yaml
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
        print(f"\nDataset classes: {data_config.get('names', [])}")
    
    # Training configuration
    print("\nTraining Configuration:")
    print(f"Dataset: {data_yaml}")
    print(f"Classes: {data_config.get('names', [])}")
    print(f"Epochs: 50")
    print(f"Image size: 640")
    print(f"Batch size: 16")
    
    # Confirm training
    confirm = input("\nStart training? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Training cancelled.")
        return
    
    # Train
    print("\nStarting training...")
    print("This will take 30-60 minutes on GPU")
    
    results = model.train(
        data=data_yaml,
        epochs=50,
        imgsz=640,
        batch=16,
        device=device,
        project=models_dir,
        name="helmet_detector_kaggle",
        exist_ok=True,
        verbose=True,
        patience=15,
        save=True,
        save_period=10,
        workers=4,
        pretrained=True,
        optimizer='auto',
        cos_lr=True,
        warmup_epochs=3
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    # Show model path
    model_path = os.path.join(models_dir, "helmet_detector_kaggle", "weights", "best.pt")
    print(f"\nModel saved to: {model_path}")
    
    # Quick validation
    print("\nRunning validation on test set...")
    val_results = model.val(data=data_yaml, split='test')
    print(f"Validation mAP50: {val_results.box.map50:.3f}")
    print(f"Validation mAP50-95: {val_results.box.map:.3f}")

if __name__ == "__main__":
    train_helmet_model()