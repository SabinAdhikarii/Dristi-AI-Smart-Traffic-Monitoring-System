import numpy as np

class SimpleTracker:
    """
    Simple IoU-based tracker for vehicle tracking.
    """
    def __init__(self, max_age=30, min_hits=3):
        self.max_age = max_age  # Frames to keep lost track
        self.min_hits = min_hits  # Minimum detections to confirm track
        self.trackers = {}
        self.next_id = 1
        self.frame_count = 0
    
    def iou(self, box1, box2):
        """Calculate Intersection over Union."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        if box1_area + box2_area - inter_area == 0:
            return 0
        
        return inter_area / float(box1_area + box2_area - inter_area)
    
    def update(self, detections):
        """
        Update tracks with new detections.
        detections: List of [x1, y1, x2, y2, confidence, class_id]
        Returns: List of [x1, y1, x2, y2, confidence, class_id, track_id]
        """
        self.frame_count += 1
        
        # Update existing tracks age
        for track_id in list(self.trackers.keys()):
            if 'age' in self.trackers[track_id]:
                self.trackers[track_id]['age'] += 1
            else:
                self.trackers[track_id]['age'] = 1
            
            # Remove old tracks
            if self.trackers[track_id]['age'] > self.max_age:
                del self.trackers[track_id]
        
        # If no detections, return empty list
        if not detections:
            return []
        
        # Assign detections to tracks
        matched_dets = []
        unmatched_dets = list(range(len(detections)))
        
        if self.trackers and detections:
            # Simple greedy matching based on IoU
            track_ids = list(self.trackers.keys())
            
            for i, track_id in enumerate(track_ids):
                if not unmatched_dets:
                    break
                    
                track_box = self.trackers[track_id]['bbox']
                best_iou = 0
                best_match_idx = -1
                
                # Find best matching detection
                for j in unmatched_dets:
                    det_box = detections[j][:4]
                    iou_score = self.iou(track_box, det_box)
                    
                    if iou_score > best_iou:
                        best_iou = iou_score
                        best_match_idx = j
                
                # Match if IoU is good enough
                if best_iou > 0.3 and best_match_idx != -1:  # IoU threshold
                    # Update track
                    self.trackers[track_id]['bbox'] = detections[best_match_idx][:4]
                    self.trackers[track_id]['age'] = 0
                    self.trackers[track_id]['hits'] = self.trackers[track_id].get('hits', 0) + 1
                    
                    # Add track_id to detection
                    matched_det = list(detections[best_match_idx]) + [track_id]
                    matched_dets.append(matched_det)
                    
                    # Remove matched detection
                    unmatched_dets.remove(best_match_idx)
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            track_id = self.next_id
            self.next_id += 1
            
            self.trackers[track_id] = {
                'bbox': detections[det_idx][:4],
                'age': 0,
                'hits': 1
            }
            
            # Add track_id to detection
            matched_det = list(detections[det_idx]) + [track_id]
            matched_dets.append(matched_det)
        
        # Only return confirmed tracks (with enough hits)
        confirmed_dets = []
        for det in matched_dets:
            track_id = det[-1]
            if self.trackers[track_id]['hits'] >= self.min_hits:
                confirmed_dets.append(det)
        
        return confirmed_dets