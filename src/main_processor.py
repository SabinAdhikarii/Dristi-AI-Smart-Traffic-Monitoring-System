"""
Main Traffic Monitoring System - GPU Optimized with Database Storage
Includes Red Light, Helmet, Speed Detection, License Plate OCR,
and Violation Clip Saving (5-second clips per violation)
"""
import os
import json
import cv2
import torch
import subprocess
from datetime import datetime
from ultralytics import YOLO
from db import save_violation, get_db_connection
from plate_detector import LicensePlateDetector

# Check GPU
DEVICE = 0 if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU (slower)")


def _dedup_violations(violation_list, frame_window=30, x_tolerance=160):
    if not violation_list:
        return violation_list

    sorted_v = sorted(violation_list, key=lambda v: v.get('frame', 0))
    kept = []

    for candidate in sorted_v:
        c_frame = candidate.get('frame', 0)
        c_track = candidate.get('track_id')
        is_duplicate = False

        for kept_v in kept:
            k_frame = kept_v.get('frame', 0)
            k_track = kept_v.get('track_id')

            if c_track and k_track and c_track == k_track:
                is_duplicate = True
                break

            if abs(c_frame - k_frame) <= frame_window:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(candidate)

    return kept


def save_violation_clip(video_path, violation_frame, fps, session_dir,
                        clip_filename, pre_seconds=2, post_seconds=3):
    """
    Extract a 5-second clip around a violation frame.
    Saves with mp4v first, then re-encodes to H264 using ffmpeg for browser playback.
    Returns clip_filename on success, None on failure.
    """
    try:
        clips_dir = os.path.join(session_dir, "violation_clips")
        os.makedirs(clips_dir, exist_ok=True)

        clip_path = os.path.join(clips_dir, clip_filename)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  [CLIP] Could not open video: {video_path}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        start_frame = max(0, violation_frame - int(pre_seconds * fps))
        end_frame   = min(total_frames - 1, violation_frame + int(post_seconds * fps))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out    = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))

        if not out.isOpened():
            print(f"  [CLIP] VideoWriter failed to open for {clip_filename}")
            cap.release()
            return None

        current = start_frame
        while current <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            current += 1

        cap.release()
        out.release()

        # Check file was written
        size = os.path.getsize(clip_path) if os.path.exists(clip_path) else 0
        if size == 0:
            print(f"  [CLIP] Warning: clip file is empty — {clip_filename}")
            return None

        # Re-encode to H264 using ffmpeg so browser can play it
        h264_path = clip_path.replace('.mp4', '_h264.mp4')
        try:
            result = subprocess.run(
                ['ffmpeg', '-y', '-i', clip_path,
                 '-vcodec', 'libx264', '-acodec', 'aac',
                 '-preset', 'fast', '-crf', '23',
                 h264_path],
                capture_output=True, timeout=60
            )
            if os.path.exists(h264_path) and os.path.getsize(h264_path) > 0:
                os.remove(clip_path)
                os.rename(h264_path, clip_path)
                print(f"  [CLIP] Saved + re-encoded: violation_clips/{clip_filename} "
                      f"(frames {start_frame}–{end_frame})")
            else:
                print(f"  [CLIP] ffmpeg failed, keeping mp4v version: {clip_filename}")
        except Exception as e:
            print(f"  [CLIP] ffmpeg error: {e} — keeping mp4v version")

        return clip_filename

    except Exception as e:
        print(f"  [CLIP] Error saving {clip_filename}: {e}")
        import traceback
        traceback.print_exc()
        return None


class TrafficMonitoringSystem:
    def __init__(self):
        print("Initializing Traffic Monitoring System")

        self.vehicle_model_path       = "models/yolov8n_trained/weights/best.pt"
        self.traffic_light_model_path = "models/traffic_light_detector/weights/best.pt"
        self.helmet_model_path        = "models/helmet_detector_kaggle/weights/best.pt"

        self.uploads_dir  = "data/uploaded_videos"
        self.results_base = "results"
        os.makedirs(self.uploads_dir, exist_ok=True)
        os.makedirs(self.results_base, exist_ok=True)

        self.vehicle_model       = None
        self.traffic_light_model = None
        self.helmet_model        = None
        self.plate_detector      = None

        self.speed_limit                  = 60
        self.calibration_pixels_per_meter = 10

        self.load_calibration()

    def load_calibration(self):
        config_path = "speed_config.json"
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                self.speed_limit                  = config.get('speed_limit', 60)
                self.calibration_pixels_per_meter = config.get('pixels_per_meter', 10)
                print(f"Loaded calibration: {self.calibration_pixels_per_meter} px/m, "
                      f"limit: {self.speed_limit} km/h")
            except Exception:
                pass

    def save_calibration(self):
        config = {
            'speed_limit':      self.speed_limit,
            'pixels_per_meter': self.calibration_pixels_per_meter,
            'calibration_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        try:
            with open("speed_config.json", 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Could not save calibration: {e}")

    def calculate_speed(self, track_id, center_x, center_y, frame_count, fps):
        if not hasattr(self, 'vehicle_positions'):
            self.vehicle_positions = {}

        if track_id not in self.vehicle_positions:
            self.vehicle_positions[track_id] = []

        self.vehicle_positions[track_id].append((frame_count, center_x, center_y))

        if len(self.vehicle_positions[track_id]) > 30:
            self.vehicle_positions[track_id].pop(0)

        if len(self.vehicle_positions[track_id]) < 2:
            return None

        first_frame, first_x, first_y = self.vehicle_positions[track_id][0]
        last_frame,  last_x,  last_y  = self.vehicle_positions[track_id][-1]

        frame_diff = last_frame - first_frame
        if frame_diff == 0:
            return None

        pixel_distance  = ((last_x - first_x) ** 2 + (last_y - first_y) ** 2) ** 0.5
        distance_meters = pixel_distance / self.calibration_pixels_per_meter
        time_seconds    = frame_diff / fps
        speed_mps       = distance_meters / time_seconds if time_seconds > 0 else 0
        return speed_mps * 3.6

    def load_models(self):
        print("\nLoading models...")
        self.vehicle_model       = YOLO(self.vehicle_model_path)
        self.traffic_light_model = YOLO(self.traffic_light_model_path)
        self.helmet_model        = YOLO(self.helmet_model_path)

        if DEVICE != 'cpu':
            self.vehicle_model.to('cuda')
            self.traffic_light_model.to('cuda')
            self.helmet_model.to('cuda')
            print("Models moved to GPU")

        self.plate_detector = LicensePlateDetector()
        print("All models loaded successfully")

    def process_video(self, video_path, video_filename, session_id=None, calibrate=False):

        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        session_dir = os.path.join(self.results_base, session_id)
        os.makedirs(session_dir, exist_ok=True)

        print(f"\nProcessing video : {video_filename}")
        print(f"Session ID       : {session_id}")
        print(f"Device           : {'GPU' if DEVICE != 'cpu' else 'CPU'}")

        self.load_models()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None

        fps          = int(cap.get(cv2.CAP_PROP_FPS))
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")

        ret, first_frame = cap.read()
        if not ret:
            print("Error: Could not read first frame")
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        output_video_path = os.path.join(session_dir, f"{video_filename}_analyzed.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out    = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        screenshots_dir = os.path.join(session_dir, "screenshots")
        plates_dir      = os.path.join(session_dir, "license_plates")
        os.makedirs(screenshots_dir, exist_ok=True)
        os.makedirs(plates_dir,      exist_ok=True)

        frame_count         = 0
        current_light_state = None
        light_confidence    = 0
        violations = {'red_light': [], 'speeding': [], 'helmet': []}
        pending_clips = []

        self.vehicle_positions = {}
        recorded_speeding      = set()
        recorded_red_light     = set()
        spatial_red_light      = set()
        recorded_helmet        = {}

        X_ZONE_GRID     = 160
        HELMET_COOLDOWN = 90
        stop_line_y       = int(height * 0.7)
        stop_line_x_start = 0
        stop_line_x_end   = width
        red_light_active  = False

        print("\nProcessing video frames...")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {frame_count}/{total_frames} ({progress:.1f}%)")

                if frame_count % 10 == 0:
                    light_results = self.traffic_light_model(
                        frame, conf=0.3, device=DEVICE, verbose=False
                    )
                    for r in light_results:
                        if r.boxes is not None:
                            for box in r.boxes:
                                cls_id      = int(box.cls[0])
                                conf        = float(box.conf[0])
                                light_state = self.traffic_light_model.names[cls_id]
                                if conf > light_confidence:
                                    light_confidence    = conf
                                    current_light_state = light_state

                is_red = current_light_state in ['red', 'redyellow', 'unknown']

                if is_red and not red_light_active:
                    red_light_active = True
                elif not is_red and red_light_active:
                    recorded_red_light = set()
                    spatial_red_light  = set()
                    red_light_active   = False

                vehicle_results = self.vehicle_model.track(
                    frame, persist=True, device=DEVICE, verbose=False, conf=0.4
                )
                helmet_results = self.helmet_model(
                    frame, conf=0.4, device=DEVICE, verbose=False
                )

                annotated = frame.copy()
                cv2.line(annotated, (stop_line_x_start, stop_line_y),
                         (stop_line_x_end, stop_line_y), (0, 255, 0), 2)
                cv2.putText(annotated, "STOP LINE",
                            (stop_line_x_start + 10, stop_line_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                if vehicle_results[0].boxes is not None:
                    for box in vehicle_results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        cls_id       = int(box.cls[0])
                        conf         = float(box.conf[0])
                        vehicle_type = self.vehicle_model.names[cls_id]

                        track_id = None
                        if hasattr(box, 'id') and box.id is not None:
                            track_id = int(box.id[0])
                        else:
                            center_x_tmp = (x1 + x2) // 2
                            track_id = f"pos_{center_x_tmp // 50}_{frame_count // 30}"

                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        bottom_y = y2
                        x_zone   = center_x // X_ZONE_GRID

                        speed = self.calculate_speed(
                            track_id, center_x, center_y, frame_count, fps
                        )

                        if speed is not None and speed > self.speed_limit:
                            print(f"  [SPEED] Vehicle {track_id} at {speed:.1f} km/h")
                            if track_id not in recorded_speeding:
                                recorded_speeding.add(track_id)
                                vid           = len(violations['speeding']) + 1
                                ss            = os.path.join(screenshots_dir, f"speeding_{vid}.jpg")
                                clip_filename = f"speeding_{vid}.mp4"
                                speed_ann = annotated.copy()
                                cv2.putText(speed_ann,
                                            f"SPEEDING: {speed:.1f} km/h (Limit: {self.speed_limit})",
                                            (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                cv2.imwrite(ss, speed_ann)
                                pending_clips.append({
                                    'type': 'speeding', 'index': vid,
                                    'frame': frame_count, 'filename': clip_filename
                                })
                                violations['speeding'].append({
                                    'id': vid, 'frame': frame_count,
                                    'timestamp': frame_count / fps,
                                    'vehicle_type': vehicle_type, 'confidence': conf,
                                    'track_id': str(track_id),
                                    'speed': round(speed, 1), 'speed_limit': self.speed_limit,
                                    'screenshot': f"speeding_{vid}.jpg", 'clip': clip_filename
                                })
                                save_violation(session_id, 'speeding', vehicle_type,
                                               frame_count / fps, frame_count, conf, ss, video_filename)
                                print(f"  [DB] Speeding saved: {vehicle_type} @ {speed:.1f} km/h")

                        if is_red and bottom_y > stop_line_y:
                            if stop_line_x_start <= center_x <= stop_line_x_end:
                                already_by_id   = (track_id is not None and track_id in recorded_red_light)
                                already_by_zone = (x_zone in spatial_red_light)
                                if not already_by_id and not already_by_zone:
                                    if track_id is not None:
                                        recorded_red_light.add(track_id)
                                    spatial_red_light.add(x_zone)
                                    vid           = len(violations['red_light']) + 1
                                    ss            = os.path.join(screenshots_dir, f"red_light_{vid}.jpg")
                                    clip_filename = f"red_light_{vid}.mp4"
                                    plate_text, plate_confidence, plate_image_path, plate_box = None, 0, None, None
                                    try:
                                        plate_result = self.plate_detector.process_vehicle(frame, (x1, y1, x2, y2))
                                        if plate_result:
                                            plate_text       = plate_result['plate_text']
                                            plate_confidence = plate_result['confidence']
                                            plate_box        = plate_result['plate_box']
                                            plate_image_path = self.plate_detector.save_plate_image(
                                                frame, plate_box, plate_text, vid, session_dir)
                                            print(f"  [PLATE] {plate_text} (conf: {plate_confidence:.2f})")
                                            px1, py1, px2, py2 = plate_box
                                            cv2.rectangle(annotated, (px1, py1), (px2, py2), (255, 255, 0), 2)
                                            cv2.putText(annotated, plate_text, (px1, py1 - 5),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                                        else:
                                            print(f"  [PLATE] No plate for vehicle {track_id}")
                                    except Exception as e:
                                        print(f"  [PLATE] Error: {e}")
                                    cv2.imwrite(ss, annotated)
                                    pending_clips.append({
                                        'type': 'red_light', 'index': vid,
                                        'frame': frame_count, 'filename': clip_filename
                                    })
                                    violations['red_light'].append({
                                        'id': vid, 'frame': frame_count,
                                        'timestamp': frame_count / fps,
                                        'vehicle_type': vehicle_type, 'confidence': conf,
                                        'track_id': str(track_id),
                                        'screenshot': f"red_light_{vid}.jpg", 'clip': clip_filename,
                                        'license_plate': plate_text, 'plate_confidence': plate_confidence,
                                        'plate_image': plate_image_path
                                    })
                                    save_violation(session_id, 'red_light', vehicle_type,
                                                   frame_count / fps, frame_count, conf, ss, video_filename,
                                                   license_plate=plate_text,
                                                   plate_confidence=plate_confidence,
                                                   plate_image_path=plate_image_path)
                                    print(f"  [DB] Red light saved: {vehicle_type} | Plate: {plate_text or 'Not detected'}")

                        # FIX: was broken tuple unpacking — now proper conditionals
                        is_speeding_vehicle    = (speed is not None and speed > self.speed_limit)
                        is_red_light_violation = (track_id in recorded_red_light)
                        is_violation = is_speeding_vehicle or is_red_light_violation
                        color     = (0, 0, 255) if is_violation else (0, 255, 0)
                        thickness = 3 if is_violation else 2

                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
                        label = f"{vehicle_type} {conf:.2f} | {speed:.1f} km/h" if speed is not None else f"{vehicle_type} {conf:.2f}"

                        if is_speeding_vehicle:
                            vl = f"SPEEDING: {speed:.1f} km/h (Limit: {self.speed_limit})"
                            (tw, th), _ = cv2.getTextSize(vl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(annotated, (x1, y1 - th - 25), (x1 + tw + 10, y1 - 20), (0, 0, 255), -1)
                            cv2.putText(annotated, vl, (x1 + 5, y1 - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        elif is_red_light_violation:
                            vl = "RED LIGHT VIOLATION"
                            (tw, th), _ = cv2.getTextSize(vl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(annotated, (x1, y1 - th - 25), (x1 + tw + 10, y1 - 20), (0, 0, 255), -1)
                            cv2.putText(annotated, vl, (x1 + 5, y1 - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if helmet_results[0].boxes is not None:
                    for box in helmet_results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        cls_id     = int(box.cls[0])
                        conf       = float(box.conf[0])
                        class_name = self.helmet_model.names[cls_id]
                        rider_zone = (x1 // X_ZONE_GRID, y1 // X_ZONE_GRID)
                        is_no_helmet = ('without_helmet' in class_name.lower() or
                                        'nohelmet' in class_name.lower())

                        if is_no_helmet:
                            last_seen = recorded_helmet.get(rider_zone, -HELMET_COOLDOWN)
                            if (frame_count - last_seen) >= HELMET_COOLDOWN:
                                recorded_helmet[rider_zone] = frame_count
                                vid           = len(violations['helmet']) + 1
                                ss            = os.path.join(screenshots_dir, f"helmet_{vid}.jpg")
                                clip_filename = f"helmet_{vid}.mp4"
                                cv2.imwrite(ss, annotated)
                                pending_clips.append({
                                    'type': 'helmet', 'index': vid,
                                    'frame': frame_count, 'filename': clip_filename
                                })
                                violations['helmet'].append({
                                    'id': vid, 'frame': frame_count,
                                    'timestamp': frame_count / fps,
                                    'class': class_name, 'confidence': conf,
                                    'screenshot': f"helmet_{vid}.jpg", 'clip': clip_filename
                                })
                                save_violation(session_id, 'helmet', class_name,
                                               frame_count / fps, frame_count, conf, ss, video_filename)
                                print(f"  [DB] Helmet violation saved: {class_name}")
                            color, thickness = (0, 0, 255), 3
                            label = f"NO HELMET {conf:.2f}"
                            vl = "HELMET VIOLATION"
                            (tw, th), _ = cv2.getTextSize(vl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(annotated, (x1, y1 - th - 25), (x1 + tw + 10, y1 - 20), (0, 0, 255), -1)
                            cv2.putText(annotated, vl, (x1 + 5, y1 - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        else:
                            color, thickness = (0, 255, 0), 2
                            label = f"HELMET {conf:.2f}"
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
                        cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if current_light_state:
                    light_color = (0, 0, 255) if is_red else (0, 255, 0)
                    cv2.putText(annotated,
                                f"TRAFFIC LIGHT: {current_light_state.upper()} ({light_confidence:.2f})",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, light_color, 2)
                cv2.putText(annotated, f"Speed Limit: {self.speed_limit} km/h",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(annotated, f"Frame: {frame_count}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(annotated,
                            f"Violations: RL:{len(violations['red_light'])} "
                            f"Spd:{len(violations['speeding'])} Hlm:{len(violations['helmet'])}",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                lx = width - 180
                ly = height - 80
                cv2.rectangle(annotated, (lx - 10, ly - 70), (width - 10, height - 10), (0, 0, 0), -1)
                cv2.rectangle(annotated, (lx - 10, ly - 70), (width - 10, height - 10), (100, 100, 100), 1)
                cv2.putText(annotated, "LEGEND", (lx, ly - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.rectangle(annotated, (lx, ly - 40), (lx + 20, ly - 25), (0, 255, 0), 2)
                cv2.putText(annotated, "Normal",   (lx + 25, ly - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.rectangle(annotated, (lx, ly - 20), (lx + 20, ly - 5), (0, 0, 255), 3)
                cv2.putText(annotated, "Speeding", (lx + 25, ly - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.rectangle(annotated, (lx, ly), (lx + 20, ly + 15), (0, 0, 255), 3)
                cv2.putText(annotated, "Red Light", (lx + 25, ly + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.rectangle(annotated, (lx, ly + 20), (lx + 20, ly + 35), (0, 0, 255), 3)
                cv2.putText(annotated, "No Helmet", (lx + 25, ly + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                out.write(annotated)

        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")

        finally:
            cap.release()
            out.release()

        if pending_clips:
            print(f"\nExtracting {len(pending_clips)} violation clips...")
            for clip_info in pending_clips:
                result = save_violation_clip(
                    video_path=video_path,
                    violation_frame=clip_info['frame'],
                    fps=fps,
                    session_dir=session_dir,
                    clip_filename=clip_info['filename'],
                    pre_seconds=2,
                    post_seconds=3
                )
                if not result:
                    for v in violations[clip_info['type']]:
                        if v.get('id') == clip_info['index']:
                            v['clip'] = None
                            break

        for vtype in ['red_light', 'speeding', 'helmet']:
            violations[vtype] = _dedup_violations(violations[vtype])
            for i, v in enumerate(violations[vtype], 1):
                v['id'] = i

        total_violations = sum(len(violations[k]) for k in violations)

        results = {
            'session_id':      session_id,
            'video_filename':  video_filename,
            'video_path':      video_path,
            'processed_video': output_video_path,
            'processing_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'processing_time': frame_count / fps if fps > 0 else 0,
            'duration':        frame_count / fps if fps > 0 else 0,
            'total_frames':    frame_count,
            'violations':      violations,
            'summary': {
                'total_violations': total_violations,
                'red_light_count':  len(violations['red_light']),
                'speeding_count':   len(violations['speeding']),
                'helmet_count':     len(violations['helmet']),
            }
        }

        report_path = os.path.join(session_dir, 'final_report.json')
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*60}")
        print("PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Session ID       : {session_id}")
        print(f"Frames processed : {frame_count}")
        print(f"Total violations : {total_violations}")
        print(f"  Red light      : {len(violations['red_light'])}")
        print(f"  Speeding       : {len(violations['speeding'])}")
        print(f"  Helmet         : {len(violations['helmet'])}")
        print(f"Report saved to  : {report_path}")

        return results


def process_uploaded_video(video_path, video_filename, session_id=None, calibrate=False):
    system = TrafficMonitoringSystem()
    return system.process_video(video_path, video_filename,
                                session_id=session_id, calibrate=calibrate)


if __name__ == "__main__":
    print("main_processor.py loaded successfully")
