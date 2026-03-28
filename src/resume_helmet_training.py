from ultralytics import YOLO

# Load the last checkpoint
model = YOLO('models/helmet_detector_kaggle/weights/last.pt')

# Resume training for 10 more epochs
results = model.train(resume=True, epochs=50)
