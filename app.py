"""
Flask Web Application for Dristi-AI Traffic Monitoring System
with PostgreSQL Database Integration, Live Video Streaming, and Admin Dashboard
"""
import os
import json
import uuid
import queue
import time
import threading
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
from flask import (Flask, Response, render_template, request, redirect,
                   url_for, flash, jsonify, send_file, session)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

import sys
sys.path.insert(0, 'src')
from main_processor import process_uploaded_video
from stream_processor import StreamProcessor
from ocr_processor import PostOCRProcessor
from db import (
    get_admin_by_username, get_user_by_citizenship,
    update_user_last_login, register_user,
    save_violation, get_recent_sessions, get_admin_stats,
    get_all_officers, toggle_officer_status, delete_officer,
    get_all_violations, issue_fine, get_all_fines, update_fine_status,
    get_all_news, get_published_news, create_news, update_news, delete_news
)

DB_CONFIG = {
    'host':     os.environ.get('DB_HOST', 'localhost'),
    'database': os.environ.get('DB_NAME', 'Nepal-Traffic-DB'),
    'user':     os.environ.get('DB_USER', 'postgres'),
    'password': os.environ.get('DB_PASSWORD', 'S@bin123456'),
    'port':     os.environ.get('DB_PORT', '5432')
}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dristi-ai-secret-key-2025'
app.config['UPLOAD_FOLDER'] = 'data/uploaded_videos'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'MP4', 'AVI', 'MOV', 'MKV'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

app.jinja_env.filters['basename'] = os.path.basename

active_processing = {}
stream_processors = {}


def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {e.lower() for e in ALLOWED_EXTENSIONS}


# ── Auth decorators ────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page.', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_id' not in session:
            flash('Admin login required.', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


# ============== LOGIN / REGISTER / LOGOUT ==============

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'admin_id' in session:
        return redirect(url_for('admin_dashboard'))
    if 'user_id' in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        admin = get_admin_by_username(username)
        if admin and check_password_hash(admin['password_hash'], password):
            session['admin_id']   = admin['id']
            session['admin_name'] = admin['name']
            flash(f"Welcome, {admin['name']}!", 'success')
            return redirect(url_for('admin_dashboard'))

        user = get_user_by_citizenship(username)
        if user and user['is_active'] and check_password_hash(user['password_hash'], password):
            session['user_id']      = user['id']
            session['user_name']    = f"{user['first_name']} {user['last_name']}"
            session['badge_number'] = user['citizenship_number']
            update_user_last_login(user['id'])
            flash(f"Welcome back, {user['first_name']}!", 'success')
            return redirect(url_for('index'))

        flash('Invalid credentials. Please try again.', 'danger')

    return render_template('auth.html')


@app.route('/register', methods=['POST'])
def register():
    first_name         = request.form.get('first_name', '').strip()
    last_name          = request.form.get('last_name', '').strip()
    citizenship_number = request.form.get('citizenship_number', '').strip()
    issued_district    = request.form.get('issued_district', '').strip()
    date_of_birth      = request.form.get('date_of_birth', '').strip()
    mobile_number      = request.form.get('mobile_number', '').strip()
    email              = request.form.get('email', '').strip()
    password           = request.form.get('password', '')
    confirm_password   = request.form.get('confirm_password', '')

    if password != confirm_password:
        flash('Passwords do not match.', 'danger')
        return redirect(url_for('auth'))

    if len(password) < 8:
        flash('Password must be at least 8 characters.', 'danger')
        return redirect(url_for('auth'))

    password_hash = generate_password_hash(password)
    success, message = register_user(
        first_name, last_name, citizenship_number, issued_district,
        date_of_birth, mobile_number, email, password_hash
    )

    if success:
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    else:
        flash(f'Registration error: {message}', 'danger')
        return redirect(url_for('auth'))


@app.route('/auth')
def auth():
    if 'admin_id' in session:
        return redirect(url_for('admin_dashboard'))
    if 'user_id' in session:
        return redirect(url_for('index'))
    return render_template('auth.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


# ============== MAIN USER ROUTES ==============

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
            cur.close(); conn.close()
        except:
            pass

    total_videos = total_violations = 0
    if os.path.exists(app.config['RESULTS_FOLDER']):
        for sid in os.listdir(app.config['RESULTS_FOLDER']):
            rp = os.path.join(app.config['RESULTS_FOLDER'], sid, 'final_report.json')
            if os.path.exists(rp):
                try:
                    with open(rp) as f:
                        r = json.load(f)
                    total_videos += 1
                    total_violations += r.get('summary', {}).get('total_violations', 0)
                except:
                    pass

    recent_videos = []
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        for f in os.listdir(app.config['UPLOAD_FOLDER'])[:10]:
            recent_videos.append({
                'filename': f,
                'upload_date': datetime.fromtimestamp(
                    os.path.getctime(os.path.join(app.config['UPLOAD_FOLDER'], f))
                ).strftime('%Y-%m-%d')
            })

    return render_template('dashboard.html',
                           user_info=user_info,
                           total_videos=total_videos,
                           total_violations=total_violations,
                           recent_videos=recent_videos)


# ============== ADMIN ROUTES ==============

@app.route('/admin')
@admin_required
def admin_dashboard():
    stats    = get_admin_stats()
    officers = get_all_officers()
    return render_template('admin_dashboard.html', stats=stats, officers=officers)


@app.route('/admin/users')
@admin_required
def admin_users():
    officers = get_all_officers()
    return render_template('admin_users.html', officers=officers)


@app.route('/admin/users/toggle/<int:user_id>', methods=['POST'])
@admin_required
def admin_toggle_user(user_id):
    is_active = request.form.get('is_active') == 'true'
    if toggle_officer_status(user_id, is_active):
        flash('Officer status updated.', 'success')
    else:
        flash('Failed to update officer status.', 'danger')
    return redirect(url_for('admin_users'))


@app.route('/admin/users/delete/<int:user_id>', methods=['POST'])
@admin_required
def admin_delete_user(user_id):
    if delete_officer(user_id):
        flash('Officer deleted.', 'success')
    else:
        flash('Failed to delete officer.', 'danger')
    return redirect(url_for('admin_users'))


@app.route('/admin/violations')
@admin_required
def admin_violations():
    vtype      = request.args.get('type')
    violations = get_all_violations(limit=200, vtype=vtype)
    return render_template('admin_violations.html', violations=violations, vtype=vtype)


@app.route('/admin/fines')
@admin_required
def admin_fines():
    status = request.args.get('status')
    fines  = get_all_fines(limit=200, status=status)
    return render_template('admin_fines.html', fines=fines, status=status)


@app.route('/admin/fines/issue', methods=['POST'])
@admin_required
def admin_issue_fine():
    violation_id   = request.form.get('violation_id')
    sess_id        = request.form.get('session_id')
    license_plate  = request.form.get('license_plate', 'Unknown')
    violation_type = request.form.get('violation_type')
    fine_amount    = request.form.get('fine_amount')
    notes          = request.form.get('notes', '')

    if not all([violation_id, violation_type, fine_amount]):
        flash('Missing required fields.', 'danger')
        return redirect(url_for('admin_violations'))

    try:
        fine_amount = float(fine_amount)
    except ValueError:
        flash('Invalid fine amount.', 'danger')
        return redirect(url_for('admin_violations'))

    if issue_fine(violation_id, sess_id, license_plate,
                  violation_type, fine_amount, session['admin_id'], notes):
        flash(f'Fine of NPR {fine_amount:.0f} issued successfully.', 'success')
    else:
        flash('Failed to issue fine.', 'danger')

    return redirect(url_for('admin_violations'))


@app.route('/admin/fines/update/<int:fine_id>', methods=['POST'])
@admin_required
def admin_update_fine(fine_id):
    status = request.form.get('status')
    if update_fine_status(fine_id, status):
        flash('Fine status updated.', 'success')
    else:
        flash('Failed to update fine.', 'danger')
    return redirect(url_for('admin_fines'))


@app.route('/admin/news')
@admin_required
def admin_news():
    news_items = get_all_news()
    return render_template('admin_news.html', news_items=news_items)


@app.route('/admin/news/create', methods=['POST'])
@admin_required
def admin_create_news():
    title   = request.form.get('title', '').strip()
    content = request.form.get('content', '').strip()
    if not title or not content:
        flash('Title and content are required.', 'danger')
        return redirect(url_for('admin_news'))
    if create_news(title, content, session['admin_id']):
        flash('News article published.', 'success')
    else:
        flash('Failed to create news.', 'danger')
    return redirect(url_for('admin_news'))


@app.route('/admin/news/update/<int:news_id>', methods=['POST'])
@admin_required
def admin_update_news(news_id):
    title        = request.form.get('title', '').strip()
    content      = request.form.get('content', '').strip()
    is_published = request.form.get('is_published') == 'true'
    if update_news(news_id, title, content, is_published):
        flash('News updated.', 'success')
    else:
        flash('Failed to update news.', 'danger')
    return redirect(url_for('admin_news'))


@app.route('/admin/news/delete/<int:news_id>', methods=['POST'])
@admin_required
def admin_delete_news(news_id):
    if delete_news(news_id):
        flash('News deleted.', 'success')
    else:
        flash('Failed to delete news.', 'danger')
    return redirect(url_for('admin_news'))


@app.route('/admin/api/stats')
@admin_required
def admin_api_stats():
    return jsonify(get_admin_stats())


# ============== VIDEO UPLOAD & STREAMING ==============

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_video():
    if request.method == 'POST':
        if 'video_file' not in request.files:
            return jsonify({'success': False, 'error': 'No file selected'})

        file = request.files['video_file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})

        if file and allowed_file(file.filename):
            original_filename = secure_filename(file.filename)
            unique_id         = str(uuid.uuid4())[:8]
            filename          = f"{unique_id}_{original_filename}"
            filepath          = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            session_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = os.path.join(app.config['RESULTS_FOLDER'], session_id)
            os.makedirs(session_dir, exist_ok=True)

            frame_queue = queue.Queue(maxsize=60)
            stream_processors[session_id] = {'queue': frame_queue, 'active': True}

            def run_stream():
                processor = StreamProcessor()
                def send_frame(fd):
                    try:
                        frame_queue.put(fd, timeout=1)
                    except queue.Full:
                        pass
                processor.process_and_stream(filepath, session_id, send_frame)
                for _ in range(3):
                    try:
                        frame_queue.put({'complete': True, 'session_id': session_id}, timeout=2)
                        break
                    except queue.Full:
                        time.sleep(0.5)
                if session_id in stream_processors:
                    stream_processors[session_id]['active'] = False

            def run_analysis():
                try:
                    process_uploaded_video(filepath, original_filename, session_id=session_id)
                    print(f"[main_processor] Finished for session {session_id}")
                except Exception as e:
                    print(f"[main_processor] Error: {e}")
                    import traceback; traceback.print_exc()

            threading.Thread(target=run_stream,   daemon=True).start()
            threading.Thread(target=run_analysis, daemon=True).start()

            return jsonify({
                'success':    True,
                'session_id': session_id,
                'stream_url': f'/live_stream/{session_id}',
                'results_url': url_for('view_results', session_id=session_id),
                'status_url':  url_for('api_status', session_id=session_id),
                'message':    'Video uploaded. Analysis started.',
            })
        else:
            return jsonify({'success': False, 'error': 'Invalid file type.'})

    return render_template('upload_video.html')


@app.route('/live_stream/<session_id>')
def live_stream(session_id):
    def generate():
        waited = 0
        while session_id not in stream_processors and waited < 80:
            time.sleep(0.1); waited += 1
        if session_id not in stream_processors:
            yield "data: error\n\n"; return

        frame_queue = stream_processors[session_id]['queue']
        frame_count = 0
        while True:
            try:
                fd = frame_queue.get(timeout=30)
                if fd.get('complete'):
                    yield "data: done\n\n"; break
                if 'frame' in fd:
                    frame_count += 1
                    yield f"data:{fd.get('progress',0):.1f}|{fd.get('violations',0)}|{fd['frame']}\n\n"
            except queue.Empty:
                yield "data: timeout\n\n"; break
            except GeneratorExit:
                break
            except Exception as e:
                print(f"[SSE] Error: {e}"); yield "data: error\n\n"; break

    resp = Response(generate(), mimetype='text/event-stream')
    resp.headers['Cache-Control']     = 'no-cache'
    resp.headers['X-Accel-Buffering'] = 'no'
    resp.headers['Connection']        = 'keep-alive'
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


@app.route('/api/stream_status/<session_id>')
@login_required
def api_stream_status(session_id):
    if session_id in stream_processors:
        sp = stream_processors[session_id]
        return jsonify({'active': sp.get('active', False),
                        'progress': sp.get('progress', 0),
                        'violations': sp.get('violations', 0)})
    rp = os.path.join(app.config['RESULTS_FOLDER'], session_id, 'final_report.json')
    if os.path.exists(rp):
        with open(rp) as f: r = json.load(f)
        return jsonify({'active': False, 'progress': 100,
                        'violations': r.get('summary', {}).get('total_violations', 0)})
    return jsonify({'active': False, 'progress': 0, 'violations': 0})


@app.route('/stream_video/<session_id>')
def stream_video(session_id):
    session_dir = os.path.join(app.config['RESULTS_FOLDER'], session_id)
    def generate():
        vp = None
        for _ in range(60):
            if os.path.exists(session_dir):
                for f in os.listdir(session_dir):
                    if f.endswith('_analyzed.mp4'):
                        vp = os.path.join(session_dir, f); break
            if vp: break
            time.sleep(1)
        if not vp: return
        with open(vp, 'rb') as v:
            while True:
                chunk = v.read(1024 * 1024)
                if not chunk: break
                yield chunk
    return Response(generate(), mimetype='video/mp4')


# ============== RESULTS & FILE ROUTES ==============

@app.route('/results/<session_id>')
@login_required
def view_results(session_id):
    rp = os.path.join(app.config['RESULTS_FOLDER'], session_id, 'final_report.json')
    if not os.path.exists(rp):
        flash('Results not found. Analysis may still be in progress.', 'warning')
        return redirect(url_for('dashboard'))
    with open(rp) as f:
        results = json.load(f)
    return render_template('results.html', results=results)


@app.route('/play_video/<session_id>/<video_type>')
@login_required
def play_video(session_id, video_type):
    session_dir = os.path.join(app.config['RESULTS_FOLDER'], session_id)
    if video_type == 'original':
        rp = os.path.join(session_dir, 'final_report.json')
        if os.path.exists(rp):
            with open(rp) as f: results = json.load(f)
            vp = results.get('video_path')
            if vp and os.path.exists(vp):
                return send_file(vp, mimetype='video/mp4')
    elif video_type == 'processed':
        if os.path.exists(session_dir):
            for f in os.listdir(session_dir):
                if f.endswith('_analyzed.mp4'):
                    return send_file(os.path.join(session_dir, f), mimetype='video/mp4')
    return "Video not found", 404


@app.route('/get_screenshot/<session_id>/<violation_type>/<filename>')
@login_required
def get_screenshot(session_id, violation_type, filename):
    filename = os.path.basename(filename)
    for folder in ['screenshots', f'{violation_type}_screenshots',
                   f'{violation_type}_violations', 'helmet_violations',
                   'red_light_screenshots', 'speed_violations']:
        path = os.path.join(app.config['RESULTS_FOLDER'], session_id, folder, filename)
        if os.path.exists(path):
            return send_file(path, mimetype='image/jpeg')
    return "Image not found", 404


@app.route('/get_violation_clip/<session_id>/<path:clip_filename>')
@login_required
def get_violation_clip(session_id, clip_filename):
    # Sanitise filename — no path traversal
    clip_filename = os.path.basename(clip_filename)
    if '..' in clip_filename or '/' in clip_filename or '\\' in clip_filename:
        return "Invalid filename", 400

    clip_path = os.path.join(
        app.config['RESULTS_FOLDER'], session_id, 'violation_clips', clip_filename
    )

    if not os.path.exists(clip_path):
        print(f"[CLIP] Not found: {clip_path}")
        return "Clip not found", 404

    file_size = os.path.getsize(clip_path)

    # ── Handle HTTP Range requests so browsers can seek inside the clip ──
    range_header = request.headers.get('Range')
    if range_header:
        try:
            # Parse "bytes=start-end"
            byte_range = range_header.replace('bytes=', '').strip()
            parts = byte_range.split('-')
            start = int(parts[0]) if parts[0] else 0
            end   = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
            end   = min(end, file_size - 1)
            length = end - start + 1

            with open(clip_path, 'rb') as f:
                f.seek(start)
                data = f.read(length)

            resp = Response(
                data,
                status=206,
                mimetype='video/mp4',
                direct_passthrough=True
            )
            resp.headers['Content-Range']  = f'bytes {start}-{end}/{file_size}'
            resp.headers['Accept-Ranges']  = 'bytes'
            resp.headers['Content-Length'] = str(length)
            resp.headers['Cache-Control']  = 'no-cache'
            return resp

        except Exception as e:
            print(f"[CLIP] Range parse error: {e}")
            # Fall through to full file response

    # ── Full file response ──
    resp = Response(
        open(clip_path, 'rb').read(),
        status=200,
        mimetype='video/mp4'
    )
    resp.headers['Accept-Ranges']  = 'bytes'
    resp.headers['Content-Length'] = str(file_size)
    resp.headers['Cache-Control']  = 'no-cache'
    return resp


@app.route('/download_report/<session_id>')
@login_required
def download_report(session_id):
    rp = os.path.join(app.config['RESULTS_FOLDER'], session_id, 'final_report.json')
    if os.path.exists(rp):
        return send_file(rp, as_attachment=True, download_name=f'report_{session_id}.json')
    return "Report not found", 404


# ============== API ENDPOINTS ==============

@app.route('/api/status/<session_id>')
@login_required
def api_status(session_id):
    rp = os.path.join(app.config['RESULTS_FOLDER'], session_id, 'final_report.json')
    if os.path.exists(rp):
        with open(rp) as f:
            r = json.load(f)
        return jsonify({
            'complete':   True,
            'session_id': session_id,
            'violations': r.get('summary', {}).get('total_violations', 0)
        })
    sd = os.path.join(app.config['RESULTS_FOLDER'], session_id)
    if os.path.exists(sd):
        for f in os.listdir(sd):
            if f.endswith('_analyzed.mp4'):
                return jsonify({'complete': False, 'progress': 50, 'video_ready': True})
        return jsonify({'complete': False, 'progress': 25, 'video_ready': False})
    return jsonify({'complete': False, 'progress': 0, 'video_ready': False})


@app.route('/api/recent-analyses')
@login_required
def api_recent_analyses():
    recent = []
    if os.path.exists(app.config['RESULTS_FOLDER']):
        for sid in sorted(os.listdir(app.config['RESULTS_FOLDER']), reverse=True)[:5]:
            rp = os.path.join(app.config['RESULTS_FOLDER'], sid, 'final_report.json')
            if os.path.exists(rp):
                try:
                    with open(rp) as f: r = json.load(f)
                    recent.append({
                        'session_id':     sid,
                        'video_filename': r.get('video_filename', 'Unknown'),
                        'date':           r.get('processing_date', ''),
                        'duration':       f"{r.get('duration', 0):.0f}s",
                        'violations':     r.get('summary', {}).get('total_violations', 0)
                    })
                except: pass
    return jsonify({'recent': recent})


@app.route('/api/violations/<session_id>')
@login_required
def api_violations(session_id):
    rp = os.path.join(app.config['RESULTS_FOLDER'], session_id, 'final_report.json')
    if os.path.exists(rp):
        with open(rp) as f: return jsonify(json.load(f))
    return jsonify({'error': 'Report not found'}), 404


@app.route('/api/stats')
@login_required
def api_stats():
    total_videos = total_violations = 0
    if os.path.exists(app.config['RESULTS_FOLDER']):
        for sd in os.listdir(app.config['RESULTS_FOLDER']):
            rp = os.path.join(app.config['RESULTS_FOLDER'], sd, 'final_report.json')
            if os.path.exists(rp):
                try:
                    with open(rp) as f: r = json.load(f)
                    total_videos += 1
                    total_violations += r.get('summary', {}).get('total_violations', 0)
                except: pass
    return jsonify({
        'total_videos':     total_videos,
        'total_violations': total_violations,
        'active_sessions':  len(os.listdir(app.config['RESULTS_FOLDER']))
                            if os.path.exists(app.config['RESULTS_FOLDER']) else 0
    })


# ============== PAGE ROUTES ==============

@app.route('/reports')
@login_required
def reports():
    reports_list = []
    if os.path.exists(app.config['RESULTS_FOLDER']):
        for sd in sorted(os.listdir(app.config['RESULTS_FOLDER']), reverse=True):
            rp = os.path.join(app.config['RESULTS_FOLDER'], sd, 'final_report.json')
            if os.path.exists(rp):
                try:
                    with open(rp) as f: r = json.load(f)
                    reports_list.append({
                        'session_id': sd,
                        'date':       r.get('processing_date', 'Unknown'),
                        'video':      r.get('video_filename', 'Unknown'),
                        'violations': r.get('summary', {}).get('total_violations', 0)
                    })
                except: pass
    return render_template('reports.html', reports=reports_list)


@app.route('/news')
@login_required
def news():
    news_items = get_published_news()
    if not news_items:
        news_items = [
            {'title': 'New AI Traffic Monitoring System Launched',
             'created_at': '2025-03-14',
             'content': 'Dristi-AI system now operational at major intersections.'},
            {'title': 'Helmet Compliance Campaign',
             'created_at': '2025-03-12',
             'content': 'Increased enforcement on motorcycle helmet usage.'},
            {'title': 'Red Light Violation Fines Increased',
             'created_at': '2025-03-10',
             'content': 'New penalties effective from April 1.'},
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
                "SELECT first_name, last_name, citizenship_number, issued_district, "
                "date_of_birth, mobile_number, email, created_at FROM users WHERE id = %s",
                (session['user_id'],)
            )
            user = cur.fetchone()
            cur.close(); conn.close()
        except: pass
    if not user:
        user = {
            'first_name': session.get('user_name', 'Officer').split()[0],
            'last_name':  session.get('user_name', 'User').split()[-1],
            'citizenship_number': session.get('badge_number', 'TP001'),
            'issued_district': 'Kathmandu',
            'date_of_birth':   '1990-01-01',
            'mobile_number':   '9800000000',
            'email':           'user@traffic.gov.np',
            'created_at':      datetime.now().strftime('%Y-%m-%d')
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
        cur.execute(
            "SELECT id, first_name, last_name, citizenship_number, password_hash "
            "FROM users WHERE citizenship_number = %s", (citizenship,)
        )
        user = cur.fetchone(); cur.close(); conn.close()
        if user:
            return (f"<pre>id: {user['id']}\nname: {user['first_name']} {user['last_name']}\n"
                    f"citizenship: {user['citizenship_number']}\n"
                    f"hash: {user['password_hash'][:50]}...</pre>")
        return f"User {citizenship} not found"
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
