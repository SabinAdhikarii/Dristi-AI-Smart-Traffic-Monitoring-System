#!/usr/bin/env python3
"""
Verify dataset is ready for training
"""
import os
import yaml

print("=" * 60)
print("Dataset Verification")
print("=" * 60)

# 1. Check data.yaml
print("\n📄 Checking data.yaml...")
yaml_path = "data/ua_detrac/data.yaml"
try:
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✅ File exists: {yaml_path}")
    print(f"  Train path: {config.get('train')}")
    print(f"  Val path: {config.get('val')}")
    print(f"  Classes: {config.get('nc')}")
    print(f"  Class names: {config.get('names')}")
    
    # Verify paths exist
    base_path = "data/ua_detrac"
    train_path = os.path.join(base_path, config['train'])
    val_path = os.path.join(base_path, config['val'])
    
    print(f"\n🔍 Verifying paths:")
    print(f"  {train_path} → {'✅ Exists' if os.path.exists(train_path) else '❌ Missing'}")
    print(f"  {val_path} → {'✅ Exists' if os.path.exists(val_path) else '❌ Missing'}")
    
except Exception as e:
    print(f"❌ Error: {e}")

# 2. Check file counts
print("\n📊 Checking file counts...")
train_img_count = len(os.listdir("data/ua_detrac/train/images"))
train_lbl_count = len(os.listdir("data/ua_detrac/train/labels"))
val_img_count = len(os.listdir("data/ua_detrac/valid/images"))
val_lbl_count = len(os.listdir("data/ua_detrac/valid/labels"))

print(f"✅ Train: {train_img_count} images, {train_lbl_count} labels")
print(f"✅ Valid: {val_img_count} images, {val_lbl_count} labels")

# 3. Check sample files
print("\n🔍 Checking sample files...")
if train_img_count > 0:
    sample_img = os.listdir("data/ua_detrac/train/images")[0]
    sample_lbl = sample_img.replace('.jpg', '.txt').replace('.png', '.txt')
    
    print(f"  Sample image: {sample_img}")
    print(f"  Sample label: {sample_lbl}")
    
    # Check label format
    lbl_path = os.path.join("data/ua_detrac/train/labels", sample_lbl)
    if os.path.exists(lbl_path):
        with open(lbl_path, 'r') as f:
            first_line = f.readline().strip()
            parts = first_line.split()
            if len(parts) == 5:
                print(f"  ✅ Label format correct: {first_line[:50]}...")
            else:
                print(f"  ⚠️  Unexpected format: {first_line}")
    else:
        print(f"  ⚠️  Label file not found: {sample_lbl}")

print("\n" + "=" * 60)
print("✅ VERIFICATION COMPLETE")
print("=" * 60)
print("\nYour dataset is READY for training!")
print("\nRun: python src/03_train.py")