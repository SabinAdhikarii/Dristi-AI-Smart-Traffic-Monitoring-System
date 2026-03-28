"""
Flask Web Application for Dristi-AI Traffic Monitoring System
with PostgreSQL Database Integration, Live Video Streaming, and Post-Processing OCR
"""
import os
import json
import uuid
import queue
import cv2
import base64
import time
import threading
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
from flask import Flask, Response, render_template, request, redirect, url_for, flash, jsonify, send_file, session
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

# Import processors
import sys
sys.path.insert(0, 'src')
from main_processor import process_uploaded_video
from stream_processor import StreamProcessor
from ocr_processor import PostOCRProcessor

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'Nepal-Traffic-DB',
    'user': 'postgres',
    'password': 'S@bin123456',
    'port': '5432'
}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'
app.config['UPLOAD_FOLDER'] = 'data/uploaded_videos'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

# Allowed file extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'MP4', 'AVI', 'MOV', 'MKV'}

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Store active processing threads and stream processors
active_processing = {}
stream_processors = {}

def get_db_connection():
    """Get PostgreSQL database connection"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {ext.lower() for ext in ALLOWED_EXTENSIONS}

def login_required(f):
    """Decorator to require login for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ============== USER AUTHENTICATION ROUTES ==============

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login page"""
    if 'user_id' in session:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        conn = get_db_connection()
        if conn:
            try:
                cur = conn.cursor(cursor_factory=RealDictCursor)
                cur.execute(
                    "SELECT id, first_name, last_name, citizenship_number, password_hash, is_active FROM users WHERE citizenship_number = %s",
                    (username,)
                )
                user = cur.fetchone()
                cur.close()
                conn.close()
                
                if user and user['is_active'] and check_password_hash(user['password_hash'], password):
                    session['user_id'] = user['id']
                    session['user_name'] = f"{user['first_name']} {user['last_name']}"
                    session['badge_number'] = user['citizenship_number']
                    
                    # Update last login
                    conn2 = get_db_connection()
                    cur2 = conn2.cursor()
                    cur2.execute(
                        "UPDATE users SET last_login = NOW() WHERE id = %s",
                        (user['id'],)
                    )
                    conn2.commit()
                    cur2.close()
                    conn2.close()
                    
                    flash(f'Welcome back, {user["first_name"]}!', 'success')
                    return redirect(url_for('index'))
                else:
                    flash('Invalid citizenship number or password', 'danger')
            except Exception as e:
                flash(f'Login error: {str(e)}', 'danger')
        else:
            flash('Database connection error. Please try again.', 'danger')
    
    return render_template('auth.html')

@app.route('/register', methods=['POST'])
def register():
    """User registration"""
    if request.method == 'POST':
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        citizenship_number = request.form.get('citizenship_number')
        issued_district = request.form.get('issued_district')
        date_of_birth = request.form.get('date_of_birth')
        mobile_number = request.form.get('mobile_number')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('auth'))
        
        if len(password) < 8:
            flash('Password must be at least 8 characters', 'danger')
            return redirect(url_for('auth'))
        
        conn = get_db_connection()
        if conn:
            try:
                cur = conn.cursor()
                
                cur.execute("SELECT id FROM users WHERE citizenship_number = %s", (citizenship_number,))
                if cur.fetchone():
                    flash('Citizenship number already registered', 'danger')
                    cur.close()
                    conn.close()
                    return redirect(url_for('auth'))
                
                if email:
                    cur.execute("SELECT id FROM users WHERE email = %s", (email,))
                    if cur.fetchone():
                        flash('Email already registered', 'danger')
                        cur.close()
                        conn.close()
                        return redirect(url_for('auth'))
                
                password_hash = generate_password_hash(password)
                
                cur.execute("""
                    INSERT INTO users (
                        first_name, last_name, citizenship_number, 
                        issued_district, date_of_birth, mobile_number, 
                        email, password_hash, created_at, is_active
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), %s)
                """, (
                    first_name, last_name, citizenship_number,
                    issued_district, date_of_birth, mobile_number,
                    email if email else None, password_hash, True
                ))
                
                conn.commit()
                cur.close()
                conn.close()
                
                flash('Registration successful! Please login.', 'success')
                return redirect(url_for('login'))
                
            except Exception as e:
                flash(f'Registration error: {str(e)}', 'danger')
                return redirect(url_for('auth'))
        else:
            flash('Database connection error. Please try again.', 'danger')
            return redirect(url_for('auth'))
    
    return redirect(url_for('auth'))

@app.route('/auth')
def auth():
    """Combined login/registration page"""
    if 'user_id' in session:
        return redirect(url_for('index'))
    return render_template('auth.html')

@app.route('/logout')
def logout():
    """User logout"""
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

# ============== MAIN PAGE ROUTES ==============

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    user_info = {}
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(
                "SELECT first_name, last_name, citizenship_number, email FROM users WHERE id = %s",
                (session['user_id'],)
            )
            user_info = cur.fetchone() or {}
            cur.close()
            conn.close()
        except:
            pass
    
    total_videos = 0
    total_violations = 0
    
    if os.path.exists(app.config['RESULTS_FOLDER']):
        for session_dir in os.listdir(app.config['RESULTS_FOLDER']):
            report_path = os.path.join(app.config['RESULTS_FOLDER'], session_dir, 'final_report.json')
            if os.path.exists(report_path):
                try:
                    with open(report_path, 'r') as f:
                        report = json.load(f)
                    total_videos += 1
                    total_violations += report.get('summary', {}).get('total_violations', 0)
                except:
                    pass
    
    recent_videos = []
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        for f in os.listdir(app.config['UPLOAD_FOLDER'])[:10]:
            recent_videos.append({
                'filename': f,
                'upload_date': datetime.fromtimestamp(os.path.getctime(os.path.join(app.config['UPLOAD_FOLDER'], f))).strftime('%Y-%m-%d')
            })
    
    return render_template('dashboard.html',
                          user_info=user_info,
                          total_videos=total_videos,
                          total_violations=total_violations,
                          recent_videos=recent_videos)

# ============== VIDEO UPLOAD & STREAMING ROUTES ==============

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_video():
    """
    Upload video and launch parallel background threads:
      1. StreamProcessor — pushes annotated frames to the browser via SSE (live preview)
      2. main_processor — full accurate analysis, saves screenshots + final_report.json, triggers OCR
    """
    if request.method == 'POST':
        if 'video_file' not in request.files:
            return jsonify({'success': False, 'error': 'No file selected'})

        file = request.files['video_file']

        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})

        if file and allowed_file(file.filename):
            original_filename = secure_filename(file.filename)
            unique_id = str(uuid.uuid4())[:8]
            filename = f"{unique_id}_{original_filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = os.path.join(app.config['RESULTS_FOLDER'], session_id)
            os.makedirs(session_dir, exist_ok=True)

            # Thread 1: StreamProcessor — live browser preview
            frame_queue = queue.Queue(maxsize=30)
            stream_processors[session_id] = {
                'queue': frame_queue,
                'active': True,
            }

            def run_stream():
                processor = StreamProcessor()

                def send_frame(frame_data):
                    try:
                        frame_queue.put(frame_data, timeout=2)
                    except queue.Full:
                        pass

                processor.process_and_stream(filepath, session_id, send_frame)
                try:
                    frame_queue.put({'complete': True, 'session_id': session_id}, timeout=5)
                except queue.Full:
                    pass
                if session_id in stream_processors:
                    stream_processors[session_id]['active'] = False

            # Thread 2: main_processor — authoritative analysis & report (now includes OCR)
            def run_analysis():
                try:
                    process_uploaded_video(filepath, original_filename)
                    print(f"main_processor finished for session {session_id}")
                except Exception as e:
                    print(f"main_processor error for session {session_id}: {e}")

            threading.Thread(target=run_stream, daemon=True).start()
            threading.Thread(target=run_analysis, daemon=True).start()

            return jsonify({
                'success': True,
                'session_id': session_id,
                'stream_url': f'/live_stream/{session_id}',
                'message': 'Video uploaded. Live preview and analysis both started.',
            })
        else:
            return jsonify({'success': False, 'error': 'Invalid file type. Allowed: mp4, avi, mov, mkv'})

    return render_template('upload_video.html')


@app.route('/live_stream/<session_id>')
@login_required
def live_stream(session_id):
    """Server-Sent Events endpoint for live video frames"""
    def generate():
        print(f"Live stream requested for session: {session_id}")
        
        if session_id not in stream_processors:
            print(f"Session {session_id} not found in stream_processors")
            yield "data: error\n\n"
            return
        
        frame_queue = stream_processors[session_id]['queue']
        print(f"Connected to queue for session {session_id}")
        
        frame_count = 0
        
        while True:
            try:
                frame_data = frame_queue.get(timeout=30)
                
                if frame_data.get('complete'):
                    print(f"Stream complete for session {session_id}")
                    yield "data: done\n\n"
                    break
                
                if 'frame' in frame_data:
                    frame_count += 1
                    progress = frame_data.get('progress', 0)
                    violations = frame_data.get('violations', 0)
                    frame_b64 = frame_data['frame']
                    
                    # Format: data:progress:violations:base64_frame
                    yield f"data:{progress:.1f}:{violations}:{frame_b64}\n\n"
                    
                    if frame_count % 30 == 0:
                        print(f"Streamed {frame_count} frames for session {session_id}")
                
            except queue.Empty:
                print(f"Queue timeout for session {session_id}")
                yield "data: timeout\n\n"
                break
            except Exception as e:
                print(f"Error streaming session {session_id}: {e}")
                yield f"data: error\n\n"
                break
    
    resp = Response(generate(), mimetype='text/event-stream')
    resp.headers['Cache-Control'] = 'no-cache'
    resp.headers['X-Accel-Buffering'] = 'no'
    resp.headers['Content-Type'] = 'text/event-stream'
    return resp


@app.route('/api/stream_status/<session_id>')
@login_required
def api_stream_status(session_id):
    """Lightweight poll for current stream progress without frame data"""
    if session_id in stream_processors:
        sp = stream_processors[session_id]
        return jsonify({
            'active': sp.get('active', False),
            'progress': sp.get('progress', 0),
            'violations': sp.get('violations', 0),
        })
    
    report_path = os.path.join(app.config['RESULTS_FOLDER'], session_id, 'final_report.json')
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            report = json.load(f)
        return jsonify({
            'active': False,
            'progress': 100,
            'violations': report.get('summary', {}).get('total_violations', 0),
        })
    return jsonify({'active': False, 'progress': 0, 'violations': 0})


@app.route('/stream_video/<session_id>')
def stream_video(session_id):
    """Stream final processed video file"""
    session_dir = os.path.join(app.config['RESULTS_FOLDER'], session_id)
    
    def generate():
        video_path = None
        for _ in range(60):
            if os.path.exists(session_dir):
                for f in os.listdir(session_dir):
                    if f.endswith('_analyzed.mp4'):
                        video_path = os.path.join(session_dir, f)
                        break
            if video_path:
                break
            time.sleep(1)
        
        if not video_path:
            return
        
        with open(video_path, 'rb') as video:
            while True:
                chunk = video.read(1024 * 1024)
                if not chunk:
                    break
                yield chunk
    
    return Response(generate(), mimetype='video/mp4')

# ============== RESULTS & FILE ROUTES ==============

@app.route('/results/<session_id>')
@login_required
def view_results(session_id):
    report_path = os.path.join(app.config['RESULTS_FOLDER'], session_id, 'final_report.json')
    if not os.path.exists(report_path):
        flash('Results not found', 'danger')
        return redirect(url_for('dashboard'))
    with open(report_path, 'r') as f:
        results = json.load(f)
    return render_template('results.html', results=results)

@app.route('/play_video/<session_id>/<video_type>')
@login_required
def play_video(session_id, video_type):
    session_dir = os.path.join(app.config['RESULTS_FOLDER'], session_id)
    
    if video_type == 'original':
        report_path = os.path.join(session_dir, 'final_report.json')
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                results = json.load(f)
            video_path = results.get('video_path')
            if video_path and os.path.exists(video_path):
                return send_file(video_path, mimetype='video/mp4')
    
    elif video_type == 'processed':
        for f in os.listdir(session_dir):
            if f.endswith('_analyzed.mp4'):
                return send_file(os.path.join(session_dir, f), mimetype='video/mp4')
    
    return "Video not found", 404

@app.route('/get_screenshot/<session_id>/<violation_type>/<filename>')
@login_required
def get_screenshot(session_id, violation_type, filename):
    possible_folders = [
        f'{violation_type}_screenshots', f'{violation_type}_violations',
        'screenshots', 'helmet_violations', 'red_light_screenshots', 'speed_violations'
    ]
    for folder in possible_folders:
        path = os.path.join(app.config['RESULTS_FOLDER'], session_id, folder, filename)
        if os.path.exists(path):
            return send_file(path, mimetype='image/jpeg')
    return "Image not found", 404

@app.route('/download_report/<session_id>')
@login_required
def download_report(session_id):
    report_path = os.path.join(app.config['RESULTS_FOLDER'], session_id, 'final_report.json')
    if os.path.exists(report_path):
        return send_file(report_path, as_attachment=True, download_name=f'report_{session_id}.json')
    return "Report not found", 404

# ============== API ENDPOINTS ==============

@app.route('/api/status/<session_id>')
@login_required
def api_status(session_id):
    report_path = os.path.join(app.config['RESULTS_FOLDER'], session_id, 'final_report.json')
    if os.path.exists(report_path):
        return jsonify({'complete': True, 'session_id': session_id})
    session_dir = os.path.join(app.config['RESULTS_FOLDER'], session_id)
    if os.path.exists(session_dir):
        for f in os.listdir(session_dir):
            if f.endswith('_analyzed.mp4'):
                return jsonify({'complete': False, 'progress': 50, 'video_ready': True})
        return jsonify({'complete': False, 'progress': 25, 'video_ready': False})
    return jsonify({'complete': False, 'progress': 0, 'video_ready': False})

@app.route('/api/recent-analyses')
@login_required
def api_recent_analyses():
    recent = []
    if os.path.exists(app.config['RESULTS_FOLDER']):
        sessions = sorted(os.listdir(app.config['RESULTS_FOLDER']), reverse=True)[:5]
        for session_id in sessions:
            report_path = os.path.join(app.config['RESULTS_FOLDER'], session_id, 'final_report.json')
            if os.path.exists(report_path):
                try:
                    with open(report_path, 'r') as f:
                        report = json.load(f)
                    recent.append({
                        'session_id': session_id,
                        'video_filename': report.get('video_filename', 'Unknown'),
                        'date': report.get('processing_date', ''),
                        'duration': f"{report.get('duration', 0):.0f}s",
                        'violations': report.get('summary', {}).get('total_violations', 0)
                    })
                except:
                    pass
    return jsonify({'recent': recent})

@app.route('/api/violations/<session_id>')
@login_required
def api_violations(session_id):
    report_path = os.path.join(app.config['RESULTS_FOLDER'], session_id, 'final_report.json')
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            results = json.load(f)
        return jsonify(results)
    return jsonify({'error': 'Report not found'}), 404

@app.route('/api/stats')
@login_required
def api_stats():
    total_videos = 0
    total_violations = 0
    if os.path.exists(app.config['RESULTS_FOLDER']):
        for session_dir in os.listdir(app.config['RESULTS_FOLDER']):
            report_path = os.path.join(app.config['RESULTS_FOLDER'], session_dir, 'final_report.json')
            if os.path.exists(report_path):
                try:
                    with open(report_path, 'r') as f:
                        report = json.load(f)
                    total_videos += 1
                    total_violations += report.get('summary', {}).get('total_violations', 0)
                except:
                    pass
    return jsonify({
        'total_videos': total_videos,
        'total_violations': total_violations,
        'active_sessions': len(os.listdir(app.config['RESULTS_FOLDER'])) if os.path.exists(app.config['RESULTS_FOLDER']) else 0
    })

# ============== PAGE ROUTES ==============

@app.route('/reports')
@login_required
def reports():
    reports_list = []
    if os.path.exists(app.config['RESULTS_FOLDER']):
        for session_dir in sorted(os.listdir(app.config['RESULTS_FOLDER']), reverse=True):
            report_path = os.path.join(app.config['RESULTS_FOLDER'], session_dir, 'final_report.json')
            if os.path.exists(report_path):
                try:
                    with open(report_path, 'r') as f:
                        report = json.load(f)
                    reports_list.append({
                        'session_id': session_dir,
                        'date': report.get('processing_date', 'Unknown'),
                        'video': report.get('video_filename', 'Unknown'),
                        'violations': report.get('summary', {}).get('total_violations', 0)
                    })
                except:
                    pass
    return render_template('reports.html', reports=reports_list)

@app.route('/news')
@login_required
def news():
    news_items = [
        {'title': 'New AI Traffic Monitoring System Launched', 'date': '2025-03-14', 'summary': 'Dristi-AI system now operational at major intersections.'},
        {'title': 'Helmet Compliance Campaign', 'date': '2025-03-12', 'summary': 'Increased enforcement on motorcycle helmet usage.'},
        {'title': 'Red Light Violation Fines Increased', 'date': '2025-03-10', 'summary': 'New penalties effective from April 1.'},
        {'title': 'Speed Cameras Installed at 5 New Locations', 'date': '2025-03-08', 'summary': 'Expanding coverage to reduce accidents.'}
    ]
    return render_template('news.html', news_items=news_items)

@app.route('/profile')
@login_required
def profile():
    conn = get_db_connection()
    user = None
    if conn:
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(
                "SELECT first_name, last_name, citizenship_number, issued_district, date_of_birth, mobile_number, email, created_at FROM users WHERE id = %s",
                (session['user_id'],)
            )
            user = cur.fetchone()
            cur.close()
            conn.close()
        except:
            pass
    if not user:
        user = {
            'first_name': session.get('user_name', 'Admin').split()[0] if session.get('user_name') else 'Admin',
            'last_name': session.get('user_name', 'User').split()[-1] if session.get('user_name') and ' ' in session.get('user_name') else 'User',
            'citizenship_number': session.get('badge_number', 'TP001'),
            'issued_district': 'Kathmandu',
            'date_of_birth': '1990-01-01',
            'mobile_number': '9800000000',
            'email': 'user@traffic.gov.np',
            'created_at': datetime.now().strftime('%Y-%m-%d')
        }
    return render_template('profile.html', user=user)

@app.route('/settings')
@login_required
def settings():
    return render_template('settings.html')

# ============== DEBUG ROUTES ==============

@app.route('/debug_user/<citizenship>')
def debug_user(citizenship):
    conn = get_db_connection()
    if conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT id, first_name, last_name, citizenship_number, password_hash FROM users WHERE citizenship_number = %s", (citizenship,))
        user = cur.fetchone()
        cur.close()
        conn.close()
        if user:
            return f"<pre>id: {user['id']}\nfirst_name: {user['first_name']}\nlast_name: {user['last_name']}\ncitizenship_number: {user['citizenship_number']}\npassword_hash: {user['password_hash'][:50]}...</pre>"
        return f"User with citizenship {citizenship} not found"
    return "Database connection failed"

@app.route('/test_password/<citizenship>/<password>')
def test_password(citizenship, password):
    conn = get_db_connection()
    if conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT password_hash FROM users WHERE citizenship_number = %s", (citizenship,))
        user = cur.fetchone()
        cur.close()
        conn.close()
        if user:
            return f"Password match: {check_password_hash(user['password_hash'], password)}"
        return "User not found"
    return "DB error"

# ============== ERROR HANDLERS ==============

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)