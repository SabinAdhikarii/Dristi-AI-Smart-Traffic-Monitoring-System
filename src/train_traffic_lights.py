"""
Train Traffic Light Detection Model
This script trains a YOLOv8 model to detect traffic light states
"""

import os
from pyexpat import model
import torch
from ultralytics import YOLO

def main():
    # Print header
    print("=" * 60)
    print("TRAFFIC LIGHT DETECTION TRAINING")
    print("=" * 60)
    
    # Define paths
    project_root = os.getcwd()
    data_yaml = os.path.join(project_root, "data", "traffic_lights", "tlony", "data.yaml")
    models_dir = os.path.join(project_root, "models")
    
    # Check if dataset exists
    if not os.path.exists(data_yaml):
        print(f"ERROR: Dataset not found at {data_yaml}")
        print("Please ensure your traffic light dataset is properly set up")
        return
    
    print(f"Dataset configuration: {data_yaml}")
    
    # Check GPU availability
    if torch.cuda.is_available():
        device = 0  # Use first GPU
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU detected: {gpu_name}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = 'cpu'
        print("WARNING: No GPU detected. Training will be slow on CPU")
    
    # Load the model
    print("\nLoading YOLOv8n model...")
    try:
        model = YOLO('yolov8n.pt')  # Load pretrained model
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Display training configuration
    print("\nTraining Configuration:")
    print(f"  Dataset: {data_yaml}")
    print(f"  Number of classes: 8")
    print(f"  Class names: red, green, yellow, redyellow, unknown, pedred, pedgreen, pedunknown")
    print(f"  Epochs: 50")
    print(f"  Image size: 640")
    print(f"  Batch size: 16")
    print(f"  Device: {device}")
    print(f"  Model save location: {os.path.join(models_dir, 'traffic_light_detector')}")
    
    # Confirm before starting
    print("\n" + "-" * 40)
    response = input("Start training? (y/n): ").lower().strip()
    if response != 'y':
        print("Training cancelled")
        return
    
    # Start training
    print("\nStarting training...")
    print("This may take 30-60 minutes depending on your GPU")
    print("Press Ctrl+C to stop early\n")
    
    try:
        results = model.train(
            data=data_yaml,
            epochs=50,
            imgsz=640,
            batch=16,
            device=device,
            project=models_dir,
            name="traffic_light_detector",
            exist_ok=True,
            verbose=True,
            patience=15,           # Stop if no improvement for 15 epochs
            save=True,              # Save checkpoints
            save_period=10,         # Save every 10 epochs
            workers=4,               # Number of data loading workers
            pretrained=True,         # Use pretrained weights
            optimizer='auto',        # Automatically choose optimizer
            cos_lr=True,             # Use cosine learning rate scheduler
            warmup_epochs=3,         # Warmup epochs
            warmup_momentum=0.8,     # Warmup momentum
            warmup_bias_lr=0.1,       # Warmup bias learning rate
            label_smoothing=0.0,      # Label smoothing factor
            dropout=0.0                # Dropout rate
        )
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        # Show where model is saved
        model_path = os.path.join(models_dir, "traffic_light_detector", "weights", "best.pt")
        if os.path.exists(model_path):
            print(f"\nTrained model saved to: {model_path}")
            
            # Show model size
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"Model file size: {size_mb:.1f} MB")
            
            # Quick validation
            print("\nRunning validation on test set...")
            val_results = model.val()
            print(f"Validation mAP50: {val_results.box.map50:.3f}")
            print(f"Validation mAP50-95: {val_results.box.map:.3f}")
        else:
            print(f"\nWARNING: Model not found at expected location: {model_path}")
            print("Check the models directory for your trained model")
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Partial results may be saved in the models directory")
    
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        print("Please check your dataset and try again")
    
    print("\n" + "=" * 60)
    print("Done")
    print("=" * 60)
# After training, run validation with explicit project path
print("\nRunning validation...")
val_results = model.val(
    project="models/traffic_light_detector",
    name="traffic_lights_validation_results",
    exist_ok=True
)

if __name__ == "__main__":
    main()

