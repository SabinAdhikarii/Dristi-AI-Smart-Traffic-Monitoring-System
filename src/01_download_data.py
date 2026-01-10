#!/usr/bin/env python3
"""
Download UA-DETRAC dataset with multiple fallback options
"""
import os
import sys
import requests
import zipfile
from tqdm import tqdm
import subprocess

def setup_data_directory():
    """Create necessary directories"""
    os.makedirs('data/ua_detrac', exist_ok=True)
    os.makedirs('data/test_videos', exist_ok=True)
    print("✓ Created data directories")

def download_with_resume(url, filename):
    """Download large file with resume capability"""
    print(f"Downloading {filename} from {url}")
    
    # Check if partially downloaded
    if os.path.exists(filename):
        print(f"  Found existing file, resuming...")
        headers = {}
        file_size = os.path.getsize(filename)
        headers['Range'] = f'bytes={file_size}-'
    else:
        headers = {}
    
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        total_size = int(response.headers.get('content-length', 0))
        
        mode = 'ab' if 'Range' in headers else 'wb'
        with open(filename, mode) as file, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
        
        print(f"✓ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False

def download_via_kaggle():
    """Try downloading via Kaggle CLI"""
    print("\nTrying Kaggle download...")
    try:
        # First check if kaggle is installed
        subprocess.run(['kaggle', '--version'], check=True, capture_output=True)
        
        # Try different dataset identifiers
        datasets = [
            'ujwalkandi/ua-detrac',
            'tobyzhyua/detrac-dataset',
            'solesensei/solesensei_bdd10k'
        ]
        
        for dataset in datasets:
            print(f"  Trying dataset: {dataset}")
            try:
                cmd = f'kaggle datasets download {dataset} -p data/ --unzip'
                subprocess.run(cmd, shell=True, check=True)
                print(f"✓ Successfully downloaded via Kaggle")
                return True
            except:
                continue
        
        return False
    except FileNotFoundError:
        print("  Kaggle CLI not installed. Install with: pip install kaggle")
        print("  You also need API token from Kaggle website")
        return False

def download_sample_only():
    """Download just a small sample for testing"""
    print("\nDownloading minimal sample for testing...")
    
    # Sample video URL (from research servers)
    sample_urls = [
        "https://github.com/..../MVI_20011.mp4",  # You need to find actual URLs
        # Note: You'll need to find public sample videos
    ]
    
    # Instead, let's use a different approach
    print("For quick testing, use these alternatives:")
    print("1. Use your own traffic videos")
    print("2. Download from YouTube using yt-dlp")
    print("3. Use a smaller traffic dataset first")
    
    return False

def extract_zip(zip_path, extract_to):
    """Extract zip file"""
    print(f"Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✓ Extracted to {extract_to}")
        return True
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False

def main():
    """Main download function"""
    print("=" * 60)
    print("UA-DETRAC Dataset Downloader")
    print("=" * 60)
    
    setup_data_directory()
    
    # Option 1: Try direct download (often fails)
    train_url = "https://detrac-db.rit.albany.edu/Data/DETRAC-train-data.zip"
    test_url = "https://detrac-db.rit.albany.edu/Data/DETRAC-test-data.zip"
    
    print("\n[1] Trying direct download...")
    success = download_with_resume(train_url, "data/DETRAC-train-data.zip")
    
    if success:
        extract_zip("data/DETRAC-train-data.zip", "data/ua_detrac")
        download_with_resume(test_url, "data/DETRAC-test-data.zip")
        extract_zip("data/DETRAC-test-data.zip", "data/ua_detrac")
    else:
        # Option 2: Try Kaggle
        print("\n[2] Direct download failed, trying Kaggle...")
        if not download_via_kaggle():
            # Option 3: Manual instructions
            print("\n" + "=" * 60)
            print("MANUAL DOWNLOAD REQUIRED")
            print("=" * 60)
            print("\nThe UA-DETRAC dataset is large (12GB+) and hard to auto-download.")
            print("\nPlease download MANUALLY:")
            print("\nSTEP 1: Go to ONE of these links:")
            print("  - https://www.kaggle.com/datasets/ujwalkandi/ua-detrac")
            print("  - http://detrac-db.rit.albany.edu/download")
            print("  - Search 'UA-DETRAC Google Drive' on GitHub")
            
            print("\nSTEP 2: Download these files:")
            print("  - DETRAC-train-data.zip (8.6GB)")
            print("  - DETRAC-test-data.zip (3.5GB)")
            
            print("\nSTEP 3: Place them in 'data/' folder")
            print("\nSTEP 4: Run this script again to extract")
            
            print("\nFor QUICK TESTING, download just 2-3 videos:")
            print("1. Find MVI_20011.mp4 and MVI_20012.mp4 online")
            print("2. Place in data/ua_detrac/videos/")
            print("3. Get corresponding XML files")
            
            return
    
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE!")
    print("=" * 60)
    print("\nYour data structure should look like:")
    print("data/ua_detrac/videos/      # .mp4 files")
    print("data/ua_detrac/annotations/ # .xml files")
    print("\nNow run: python src/02_preprocess.py")

if __name__ == "__main__":
    main()