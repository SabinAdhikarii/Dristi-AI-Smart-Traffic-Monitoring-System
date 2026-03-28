"""
Video Stream Processor for Real-time Analysis
"""
import cv2
import torch
import time
import base64
from ultralytics import YOLO

DEVICE = 0 if torch.cuda.is_available() else 'cpu'


class StreamProcessor:
    def __init__(self):
        self.vehicle_model_path = "models/yolov8n_trained/weights/best.pt"
        self.traffic_light_model_path = "models/traffic_light_detector/weights/best.pt"
        self.helmet_model_path = "models/helmet_detector_kaggle/weights/best.pt"

        self.vehicle_model = None
        self.traffic_light_model = None
        self.helmet_model = None
        self.is_ready = False

    def load_models(self):
        print("Loading models for stream processor...")
        self.vehicle_model = YOLO(self.vehicle_model_path)
        self.traffic_light_model = YOLO(self.traffic_light_model_path)
        self.helmet_model = YOLO(self.helmet_model_path)

        if DEVICE != 'cpu':
            self.vehicle_model.to('cuda')
            self.traffic_light_model.to('cuda')
            self.helmet_model.to('cuda')

        self.is_ready = True
        print("Stream processor ready")

    def process_and_stream(self, video_path, session_id, send_callback):
        if not self.is_ready:
            self.load_models()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"StreamProcessor: could not open {video_path}")
            send_callback({'complete': True, 'session_id': session_id})
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_count = 0
        current_light_state = None
        light_confidence = 0
        stop_line_y = int(height * 0.7)

        # Track violations for display
        preview_red_violators = set()
        preview_helmet_violators = set()
        total_violations_display = 0

        print(f"StreamProcessor: starting live stream for session {session_id}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                progress = (frame_count / total_frames * 100) if total_frames else 0

                # Traffic light (every 10 frames)
                if frame_count % 10 == 0:
                    light_results = self.traffic_light_model(
                        frame, conf=0.3, device=DEVICE, verbose=False
                    )
                    for r in light_results:
                        if r.boxes is not None:
                            for box in r.boxes:
                                cls_id = int(box.cls[0])
                                conf = float(box.conf[0])
                                state = self.traffic_light_model.names[cls_id]
                                if conf > light_confidence:
                                    light_confidence = conf
                                    current_light_state = state

                is_red = current_light_state in ['red', 'redyellow', 'unknown']

                # Reset violators when light turns green
                if not is_red:
                    preview_red_violators.clear()

                # Vehicle detection
                vehicle_results = self.vehicle_model.track(
                    frame, persist=True, device=DEVICE, verbose=False, conf=0.4
                )

                # Helmet detection
                helmet_results = self.helmet_model(
                    frame, conf=0.4, device=DEVICE, verbose=False
                )

                # Annotate frame
                annotated = frame.copy()
                cv2.line(annotated, (0, stop_line_y), (width, stop_line_y), (0, 255, 0), 2)
                cv2.putText(annotated, "STOP LINE",
                            (10, stop_line_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Process vehicles
                if vehicle_results[0].boxes is not None:
                    for box in vehicle_results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        vehicle_type = self.vehicle_model.names[cls_id]

                        track_id = None
                        if hasattr(box, 'id') and box.id is not None:
                            track_id = int(box.id[0])

                        # Track red light violations for display
                        if is_red and y2 > stop_line_y and track_id is not None:
                            if track_id not in preview_red_violators:
                                preview_red_violators.add(track_id)
                                total_violations_display += 1

                        color = (0, 0, 255) if track_id in preview_red_violators else (0, 255, 0)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated, f"{vehicle_type} {conf:.2f}",
                                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Process helmets
                if helmet_results[0].boxes is not None:
                    for box in helmet_results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = self.helmet_model.names[cls_id]

                        is_no_helmet = ('without_helmet' in class_name.lower() or
                                        'nohelmet' in class_name.lower())

                        rider_key = f"{class_name}_{x1//50}_{y1//50}"
                        if is_no_helmet and rider_key not in preview_helmet_violators:
                            preview_helmet_violators.add(rider_key)
                            total_violations_display += 1

                        color = (0, 0, 255) if is_no_helmet else (0, 255, 0)
                        label = f"NO HELMET {conf:.2f}" if is_no_helmet else f"HELMET {conf:.2f}"

                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated, label,
                                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # HUD
                light_color = (0, 0, 255) if is_red else (0, 255, 0)
                if current_light_state:
                    cv2.putText(annotated, f"LIGHT: {current_light_state.upper()}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, light_color, 2)
                cv2.putText(annotated, f"Processing: {progress:.1f}%",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(annotated, f"Violations: {total_violations_display}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # Encode and stream
                _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
                frame_b64 = base64.b64encode(buffer).decode('utf-8')

                # Send frame with progress and violations
                send_callback({
                    'frame': frame_b64,
                    'progress': progress,
                    'violations': total_violations_display,
                    'frame_count': frame_count,
                })

                time.sleep(0.033)  # ~30 fps cap

        except Exception as e:
            print(f"StreamProcessor error: {e}")

        cap.release()
        send_callback({'complete': True, 'session_id': session_id})
        print(f"StreamProcessor: done for session {session_id}")