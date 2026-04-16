# Dristi AI – Smart Traffic Monitoring System

AI-powered real-time traffic surveillance using YOLOv8.
**Hardware target: HP Victus | RTX 4050 (6 GB VRAM) | CUDA 13.0 | 16 GB RAM**

---

## Project Structure

```
Dristi-AI-Smart-Traffic-Monitoring-System/
├── src/
│   ├── main.py                      ← Run everything from here
│   ├── vehicle_detection.py         ← Vehicle detection (YOLOv8)
│   ├── vehicle_tracker.py           ← Multi-object tracking (IoU + centroid)
│   ├── speed_detection.py           ← Vehicle speed estimation
│   ├── traffic_light_detection.py   ← Traffic light state (red/yellow/green)
│   ├── traffic_light_violation.py   ← Red-light violation detection
│   ├── helmet_detection.py          ← Helmet / no-helmet classification
│   ├── helmet_violation.py          ← Helmet violation records + snapshots
│   ├── number_plate_detection.py    ← ANPR (plate localisation + OCR)
│   └── 03_train.py                  ← Training script for vehicle detection
├── data/ua_detrac/                  ← Your dataset (already present)
├── models/                          ← Trained weights saved here
├── outputs/                         ← Processed videos + violation evidence
└── requirements.txt
```

---

## Setup (RTX 4050 + CUDA 13.0)

### Important: CUDA version note
Your **driver** is CUDA 13.0, but PyTorch ships **cu121** (12.1) runtime builds.
This is fine — NVIDIA drivers are backward-compatible. Use `cu121` PyTorch builds.

### Step 1 — Install PyTorch with GPU support

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify your GPU is detected:
```bash
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.version.cuda)"
# Expected:
# NVIDIA GeForce RTX 4050 Laptop GPU
# 12.1
```

### Step 2 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

---

## Training Vehicle Detection (RTX 4050 optimised)

Your dataset is already at `data/ua_detrac/` in YOLO format.

```bash
# Recommended: YOLOv8s (small) — best accuracy/speed on RTX 4050
python src/03_train.py

# Full custom run
python src/03_train.py --model yolov8s.pt --epochs 100 --batch 16

# Try batch=32 if VRAM allows (check with nvidia-smi during training)
python src/03_train.py --batch 32

# Evaluate a trained model without retraining
python src/03_train.py --eval-only
```

### Model choice guide for RTX 4050 (6 GB VRAM)

| Model    | VRAM (AMP) | Training time* | mAP est. |
|----------|-----------|----------------|----------|
| yolov8n  | ~2.5 GB   | ~45 min        | Good     |
| yolov8s  | ~3.5 GB   | ~1.5 hrs       | Better ✓ |
| yolov8m  | ~5.5 GB   | ~3 hrs         | Best     |
| yolov8l  | ~7 GB     | ✗ OOM          | –        |

> **Recommendation: `yolov8s`** — fits comfortably in 6 GB with AMP, good accuracy for traffic surveillance. *Times are approximate for UA-DETRAC at 100 epochs.

### Monitor GPU during training
Open a second terminal:
```bash
nvidia-smi -l 1   # refresh every second
# Watch: GPU-Util (aim for 90%+), Mem-Usage (should be < 5.5 GB)
```

---

## Running the Full Pipeline

```bash
# Webcam
python src/main.py

# Video file
python src/main.py --source video.mp4

# Save annotated output
python src/main.py --source video.mp4 --output outputs/result.mp4

# Run only specific modules (saves VRAM)
python src/main.py --source video.mp4 --modules vehicle speed helmet

# Export violation logs after session
python src/main.py --source video.mp4 --export-violations

# List available modules
python src/main.py --list-modules
```

### Available modules

| Module                    | What it does                              |
|---------------------------|-------------------------------------------|
| `vehicle`                 | Detect + classify vehicles (always on)    |
| `speed`                   | Estimate vehicle speed in km/h            |
| `traffic_light`           | Detect red / yellow / green signal        |
| `traffic_light_violation` | Flag vehicles crossing on red             |
| `helmet`                  | Check helmet compliance on motorcycles    |
| `number_plate`            | Detect + read number plates (ANPR)        |

---

## Training Other Models

Each module needs its own labelled dataset. Train them the same way:

```bash
# Helmet detection — needs dataset with 'helmet' / 'no_helmet' labels
python src/03_train.py --data-dir data/helmet_dataset --epochs 50
# Copy result: cp models/runs/vehicle_detection/weights/best.pt models/helmet_detection.pt

# Traffic light detection
python src/03_train.py --data-dir data/traffic_light_dataset --epochs 50
# Copy: models/traffic_light_detection.pt

# Number plate detection
python src/03_train.py --data-dir data/number_plate_dataset --epochs 60
# Copy: models/number_plate_detection.pt
```

Without trained weights, each module falls back to a pretrained YOLOv8n
(pipeline runs, but accuracy on those tasks will be low).

---

## Speed Calibration

```python
speed_det.calibrate_from_points(
    pixel_point_a=(100, 400),     # two road points in frame
    pixel_point_b=(400, 400),
    real_distance_metres=12.0,    # known distance (e.g. lane width = 3.5 m)
)
```

---

## Output Files

| Path                                    | Content                      |
|-----------------------------------------|------------------------------|
| `outputs/result.mp4`                    | Annotated output video       |
| `outputs/violations/helmet/`            | Helmet violation snapshots   |
| `outputs/violations/traffic_light/`     | Red-light violation snaps    |
| `outputs/helmet_violations_<ts>.csv`    | CSV violation log            |
| `outputs/helmet_violations_<ts>.json`   | JSON violation log           |
| `models/runs/vehicle_detection/`        | Full training run (plots, weights) |

---

## RTX 4050 Performance Tips

- **AMP is ON by default** — never disable it on RTX 40xx (Ada Lovelace uses Tensor Cores for float16, ~2× faster)
- **torch.compile** activates automatically with PyTorch 2.x — ~15% free speedup
- **Cache mode `ram`** is set — your 16 GB RAM can hold the entire UA-DETRAC dataset in memory, eliminating disk I/O bottleneck
- If you see `CUDA out of memory`, reduce `--batch` from 16 to 8
