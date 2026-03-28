"""
Post-processing OCR to read license plates from saved violation screenshots
"""
import os
import cv2
import numpy as np
import easyocr
import re
from db import get_db_connection

class PostOCRProcessor:
    def __init__(self):
        print("Initializing Post-OCR Processor...")
        
        # Initialize EasyOCR
        try:
            self.reader = easyocr.Reader(['en'], gpu=True)
            print("EasyOCR initialized with GPU")
        except:
            self.reader = easyocr.Reader(['en'], gpu=False)
            print("EasyOCR initialized with CPU")
        
        print("Post-OCR Processor ready")
    
    def enhance_image(self, image):
        """Enhance image for better OCR"""
        if image is None or image.size == 0:
            return None
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, h=40)
        
        # Sharpen
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # Resize if too small (makes text larger for OCR)
        if sharpened.shape[0] < 50 or sharpened.shape[1] < 150:
            scale = max(2, int(200 / sharpened.shape[1]))
            sharpened = cv2.resize(sharpened, (sharpened.shape[1] * scale, sharpened.shape[0] * scale))
        
        return sharpened
    
    def extract_plate_from_image(self, image_path):
        """Extract license plate from violation screenshot"""
        if not os.path.exists(image_path):
            return None
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        height, width = img.shape[:2]
        
        # Focus on bottom half where license plate is likely
        bottom_roi = img[height//2:height, 0:width]
        
        # Try multiple regions
        regions = []
        
        # Region 1: Bottom half (original)
        regions.append(bottom_roi)
        
        # Region 2: If vehicle is large, try lower quarter
        if height > 400:
            lower_quarter = img[3*height//4:height, 0:width]
            regions.append(lower_quarter)
        
        # Region 3: Full image as fallback
        regions.append(img)
        
        all_results = []
        
        for roi in regions:
            # Enhance image
            enhanced = self.enhance_image(roi)
            if enhanced is None:
                continue
            
            # Try multiple threshold methods
            thresholds = []
            
            # Otsu threshold
            _, thresh1 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresholds.append(thresh1)
            
            # Adaptive threshold
            thresh2 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 15, 5)
            thresholds.append(thresh2)
            
            # Binary inverse for white plates on dark background
            _, thresh3 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            thresholds.append(thresh3)
            
            for thresh in thresholds:
                try:
                    ocr_results = self.reader.readtext(thresh, detail=1, paragraph=False)
                    for (bbox, text, conf) in ocr_results:
                        # Clean text - keep only alphanumeric
                        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
                        # Filter out common OCR errors
                        if cleaned in ['LINE', 'LINE', 'LI', 'L', 'IN', 'N']:
                            continue
                        if len(cleaned) >= 3 and conf > 0.4:
                            all_results.append({
                                'text': cleaned,
                                'confidence': conf,
                                'length': len(cleaned)
                            })
                except:
                    continue
        
        if all_results:
            # Sort by confidence and length
            all_results.sort(key=lambda x: (x['confidence'], x['length']), reverse=True)
            
            # Try to find plate pattern (2-3 letters followed by 2-4 numbers)
            for result in all_results:
                text = result['text']
                # Check if it matches plate pattern: letters + numbers
                if re.match(r'^[A-Z]{2,3}[0-9]{2,4}$', text):
                    return {'text': text, 'confidence': result['confidence']}
            
            # If no pattern match, return best result
            best = all_results[0]
            return {'text': best['text'], 'confidence': best['confidence']}
        
        return None
    
    def process_session(self, session_id):
        """Process all red light violations in a session to extract license plates"""
        conn = get_db_connection()
        if not conn:
            return
        
        print(f"\nProcessing session {session_id} for license plates...")
        
        try:
            cur = conn.cursor()
            
            # Get all red light violations for this session with empty plates
            cur.execute("""
                SELECT id, screenshot_path 
                FROM violations 
                WHERE session_id = %s 
                AND violation_type = 'red_light'
                AND (license_plate IS NULL OR license_plate = '')
            """, (session_id,))
            
            violations = cur.fetchall()
            
            if not violations:
                print("  No pending red light violations to process")
                cur.close()
                conn.close()
                return
            
            print(f"  Found {len(violations)} violations to process")
            
            for violation_id, screenshot_path in violations:
                if not screenshot_path or not os.path.exists(screenshot_path):
                    print(f"  Violation {violation_id}: Screenshot not found")
                    continue
                
                print(f"  Processing violation {violation_id}...")
                
                # Extract plate from screenshot
                result = self.extract_plate_from_image(screenshot_path)
                
                if result:
                    plate_text = result['text']
                    plate_confidence = float(result['confidence'])
                    
                    print(f"    Extracted plate: {plate_text} (conf: {plate_confidence:.2f})")
                    
                    # Update database
                    cur.execute("""
                        UPDATE violations 
                        SET license_plate = %s, plate_confidence = %s
                        WHERE id = %s
                    """, (plate_text, plate_confidence, violation_id))
                else:
                    print(f"    No plate found in screenshot")
            
            conn.commit()
            cur.close()
            
        except Exception as e:
            print(f"Error processing session: {e}")
            conn.rollback()
        
        conn.close()
        print(f"Completed processing session {session_id}")


# Test the module
if __name__ == "__main__":
    processor = PostOCRProcessor()
    print("Module loaded successfully")