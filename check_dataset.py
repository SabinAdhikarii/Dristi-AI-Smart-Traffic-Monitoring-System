"""
Simple dataset verification
"""
import os
import yaml

# Check dataset
yaml_path = 'data/traffic_lights/tlony/data.yaml'
print(f"Checking: {yaml_path}")

if os.path.exists(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    print("\nDataset config:")
    print(f"  Path: {data.get('path')}")
    print(f"  Train: {data.get('train')}")
    print(f"  Val: {data.get('val')}")
    print(f"  Classes: {len(data.get('names', {}))}")
    
    # Check train.txt
    train_txt = os.path.join(data['path'], data['train'])
    if os.path.exists(train_txt):
        with open(train_txt, 'r') as f:
            lines = f.readlines()
        print(f"\nTrain file: {len(lines)} samples")
        if lines:
            print(f"  First: {lines[0].strip()}")
    else:
        print(f"\nTrain file not found: {train_txt}")
    
    # Check val.txt
    val_txt = os.path.join(data['path'], data['val'])
    if os.path.exists(val_txt):
        with open(val_txt, 'r') as f:
            lines = f.readlines()
        print(f"Val file: {len(lines)} samples")
        if lines:
            print(f"  First: {lines[0].strip()}")
    else:
        print(f"Val file not found: {val_txt}")
else:
    print(f"data.yaml not found")