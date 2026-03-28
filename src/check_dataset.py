"""
Simple dataset verification - NO EMOJIS
"""
import os
import yaml

print("=" * 50)
print("DATASET VERIFICATION")
print("=" * 50)

# Define paths
yaml_path = "data/traffic_lights/tlony/data.yaml"
print(f"Looking for: {yaml_path}")

# Check if file exists
if not os.path.exists(yaml_path):
    print("ERROR: data.yaml not found")
    exit(1)

print("✓ data.yaml found")

# Read and parse yaml
try:
    with open(yaml_path, 'r') as f:
        content = f.read()
        print("\nFile content:")
        print("-" * 30)
        print(content)
        print("-" * 30)
        
        # Reset file pointer and parse
        f.seek(0)
        data = yaml.safe_load(f)
        
    print("\nPARSED DATA:")
    print(f"path: {data.get('path')}")
    print(f"train: {data.get('train')}")
    print(f"val: {data.get('val')}")
    print(f"classes: {len(data.get('names', {}))}")
    
except Exception as e:
    print(f"ERROR parsing YAML: {e}")
    exit(1)

# Check train.txt
train_path = os.path.join(data['path'], data['train'])
print(f"\nChecking train.txt: {train_path}")
if os.path.exists(train_path):
    with open(train_path, 'r') as f:
        lines = f.readlines()
    print(f"  ✓ Found with {len(lines)} entries")
    if len(lines) > 0:
        print(f"  First entry: {lines[0].strip()}")
else:
    print(f"  ✗ NOT FOUND at: {os.path.abspath(train_path)}")

# Check val.txt
val_path = os.path.join(data['path'], data['val'])
print(f"\nChecking val.txt: {val_path}")
if os.path.exists(val_path):
    with open(val_path, 'r') as f:
        lines = f.readlines()
    print(f"  ✓ Found with {len(lines)} entries")
    if len(lines) > 0:
        print(f"  First entry: {lines[0].strip()}")
else:
    print(f"  ✗ NOT FOUND at: {os.path.abspath(val_path)}")

# Check actual image files
print("\nCHECKING IMAGE FILES:")
train_img_dir = "data/traffic_lights/tlony/train/images"
if os.path.exists(train_img_dir):
    images = [f for f in os.listdir(train_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Train images: {len(images)} files")
else:
    print(f"Train images directory not found: {train_img_dir}")

val_img_dir = "data/traffic_lights/tlony/val/images"
if os.path.exists(val_img_dir):
    images = [f for f in os.listdir(val_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Val images: {len(images)} files")
else:
    print(f"Val images directory not found: {val_img_dir}")

print("\n" + "=" * 50)
print("VERIFICATION COMPLETE")
print("=" * 50)