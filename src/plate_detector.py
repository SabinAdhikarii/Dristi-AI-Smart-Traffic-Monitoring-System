"""
License Plate Detection and OCR Module - Enhanced Version
"""
import cv2
import numpy as np
import easyocr
import re
from datetime import datetime
import os

class LicensePlateDetector:
    def __init__(self, model_path=None):
        """
        Initialize the license plate detector
        """
        print("Initializing License Plate Detector...")

        # Initialize EasyOCR reader
        try:
            self.reader = easyocr.Reader(['en'], gpu=True)
            print("EasyOCR initialized with GPU")
        except Exception as e:
            print(f"EasyOCR GPU failed: {e}, trying CPU")
            self.reader = easyocr.Reader(['en'], gpu=False)
            print("EasyOCR initialized with CPU")

        self.model = None

        if model_path and os.path.exists(model_path):
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            print(f"Loaded plate detection model from {model_path}")
        else:
            print("No plate detection model found. Using enhanced region-based detection.")

        print("License Plate Detector ready")

    def enhance_image(self, roi):
        """
        Enhance image for better OCR
        """
        if roi.size == 0:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Increase contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, h=30)
        
        # Sharpen
        kernel = np.array([[-1,-1,-1],
                           [-1, 9,-1],
                           [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # Resize if too small
        if sharpened.shape[0] < 40 or sharpened.shape[1] < 100:
            scale = max(2, int(100 / sharpened.shape[1]))
            sharpened = cv2.resize(sharpened, (sharpened.shape[1] * scale, sharpened.shape[0] * scale))
        
        return sharpened

    def detect_plates_region(self, image, vehicle_box):
        """
        Enhanced license plate region detection
        """
        x1, y1, x2, y2 = vehicle_box
        # Expand the vehicle region to capture more of the vehicle
        margin_x = int((x2 - x1) * 0.1)
        margin_y = int((y2 - y1) * 0.2)
        x1_exp = max(0, x1 - margin_x)
        y1_exp = max(0, y1 - margin_y)
        x2_exp = min(image.shape[1], x2 + margin_x)
        y2_exp = min(image.shape[0], y2 + margin_y)
        
        vehicle_roi = image[y1_exp:y2_exp, x1_exp:x2_exp].copy()

        if vehicle_roi.size == 0:
            return []

        # Convert to grayscale
        gray = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
        
        # Multiple edge detection methods
        edges1 = cv2.Canny(gray, 50, 150)
        edges2 = cv2.Canny(gray, 100, 200)
        edges = cv2.addWeighted(edges1, 0.5, edges2, 0.5, 0)
        
        # Morphological operations to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        plates = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Skip very small or very large regions
            if w < 40 or h < 15 or w > vehicle_roi.shape[1] * 0.8:
                continue
            
            # Calculate aspect ratio
            aspect_ratio = w / h if h > 0 else 0
            
            # License plates are typically 2:1 to 5:1 ratio
            if 1.8 < aspect_ratio < 5.5:
                # Calculate area ratio relative to vehicle
                area = w * h
                vehicle_area = vehicle_roi.shape[0] * vehicle_roi.shape[1]
                area_ratio = area / vehicle_area
                
                # Check if this is likely a license plate
                if area_ratio > 0.02 and area_ratio < 0.25:
                    # Adjust coordinates back to original image
                    plate_x1 = x1_exp + x
                    plate_y1 = y1_exp + y
                    plate_x2 = x1_exp + x + w
                    plate_y2 = y1_exp + y + h
                    
                    # Calculate confidence based on aspect ratio and area
                    confidence = 0.6 + (abs(aspect_ratio - 3) / 5) * 0.3
                    confidence = min(0.9, max(0.5, confidence))
                    
                    plates.append({
                        'box': (plate_x1, plate_y1, plate_x2, plate_y2),
                        'confidence': confidence,
                        'area_ratio': area_ratio,
                        'aspect_ratio': aspect_ratio
                    })

        # Sort by confidence and area ratio
        plates.sort(key=lambda p: p['confidence'] * p['area_ratio'], reverse=True)
        
        # Debug output
        if plates:
            print(f"  [PLATE DEBUG] Found {len(plates)} plate candidates")
            for i, p in enumerate(plates[:2]):
                print(f"    Candidate {i+1}: area={p['area_ratio']:.3f}, aspect={p['aspect_ratio']:.2f}, conf={p['confidence']:.2f}")
        
        return plates[:2]  # Return top 2 candidates

    def extract_plate_text(self, image, plate_box):
        """
        Enhanced text extraction with multiple preprocessing strategies
        """
        x1, y1, x2, y2 = plate_box
        plate_roi = image[y1:y2, x1:x2]

        if plate_roi.size == 0:
            return None

        # Enhance image
        enhanced = self.enhance_image(plate_roi)
        if enhanced is None:
            return None

        # Try multiple thresholding methods
        results = []
        
        # Method 1: Otsu threshold
        _, thresh1 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Method 2: Adaptive threshold
        thresh2 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 15, 5)
        
        # Method 3: Binary inverse
        _, thresh3 = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Try each method
        for thresh, method_name in [(thresh1, "otsu"), (thresh2, "adaptive"), (thresh3, "binary_inv")]:
            try:
                ocr_results = self.reader.readtext(thresh, paragraph=False, detail=1)
                
                for (bbox, text, confidence) in ocr_results:
                    # Clean text - keep only alphanumeric
                    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
                    if len(cleaned) >= 4 and confidence > 0.3:
                        results.append({
                            'text': cleaned,
                            'confidence': confidence,
                            'method': method_name
                        })
                        print(f"  [OCR] {method_name}: '{cleaned}' (conf: {confidence:.2f})")
            except Exception as e:
                print(f"  [OCR] {method_name} failed: {e}")
                continue

        if results:
            # Group by text and take highest confidence
            best_by_text = {}
            for r in results:
                if r['text'] not in best_by_text or r['confidence'] > best_by_text[r['text']]['confidence']:
                    best_by_text[r['text']] = r
            
            # Return the best result
            best = max(best_by_text.values(), key=lambda x: x['confidence'])
            return best

        return None

    def process_vehicle(self, image, vehicle_box):
        """
        Main function to detect plate and extract text from a vehicle
        """
        # Step 1: Detect license plate region
        plates = self.detect_plates_region(image, vehicle_box)

        if not plates:
            print(f"  [PLATE] No plate candidates found for vehicle")
            return None

        # Try each plate candidate
        for plate in plates:
            # Step 2: Extract text using OCR
            plate_text = self.extract_plate_text(image, plate['box'])
            
            if plate_text and len(plate_text['text']) >= 4:
                print(f"  [PLATE] Successfully extracted: '{plate_text['text']}' (conf: {plate_text['confidence']:.2f})")
                return {
                    'plate_box': plate['box'],
                    'plate_text': plate_text['text'],
                    'confidence': plate_text['confidence'],
                    'detection_confidence': plate['confidence']
                }
            else:
                print(f"  [PLATE] OCR failed for candidate (text length: {len(plate_text['text']) if plate_text else 0})")

        return None

    def save_plate_image(self, image, plate_box, plate_text, violation_id, session_dir):
        """
        Save cropped license plate image
        """
        x1, y1, x2, y2 = plate_box
        plate_roi = image[y1:y2, x1:x2]

        if plate_roi.size == 0:
            return None
        
        # Create plates directory
        plates_dir = os.path.join(session_dir, "license_plates")
        os.makedirs(plates_dir, exist_ok=True)

        # Clean plate text for filename
        clean_text = re.sub(r'[^A-Z0-9]', '', plate_text)
        
        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plate_{violation_id}_{clean_text}_{timestamp}.jpg"
        filepath = os.path.join(plates_dir, filename)

        # Enhance for saved image
        enhanced = self.enhance_image(plate_roi)
        if enhanced is not None:
            cv2.imwrite(filepath, enhanced)
        else:
            cv2.imwrite(filepath, plate_roi)

        return filepath

# Test the module
if __name__ == "__main__":
    print("Testing License Plate Detector...")
    detector = LicensePlateDetector()
    print("Module loaded successfully")