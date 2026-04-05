"""
Database Configuration and Helper Functions
"""
import psycopg2
from psycopg2.extras import RealDictCursor

DB_CONFIG = {
    'host': 'localhost',
    'database': 'Nepal-Traffic-DB',
    'user': 'postgres',
    'password': 'S@bin123456',
    'port': '5432'
}


def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None


# ── Admin Authentication ───────────────────────────────────────────────────────

def get_admin_by_username(username):
    conn = get_db_connection()
    if not conn:
        return None
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM admins WHERE username = %s", (username,))
        admin = cur.fetchone()
        cur.close()
        conn.close()
        return admin
    except Exception as e:
        print(f"Error fetching admin: {e}")
        return None


# ── User Authentication ────────────────────────────────────────────────────────

def get_user_by_citizenship(citizenship_number):
    conn = get_db_connection()
    if not conn:
        return None
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT id, first_name, last_name, citizenship_number, password_hash, is_active FROM users WHERE citizenship_number = %s",
            (citizenship_number,)
        )
        user = cur.fetchone()
        cur.close()
        conn.close()
        return user
    except Exception as e:
        print(f"Error fetching user: {e}")
        return None


def update_user_last_login(user_id):
    conn = get_db_connection()
    if not conn:
        return
    try:
        cur = conn.cursor()
        cur.execute("UPDATE users SET last_login = NOW() WHERE id = %s", (user_id,))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error updating last login: {e}")


def register_user(first_name, last_name, citizenship_number, issued_district,
                  date_of_birth, mobile_number, email, password_hash):
    conn = get_db_connection()
    if not conn:
        return False, "Database connection error"
    try:
        cur = conn.cursor()

        cur.execute("SELECT id FROM users WHERE citizenship_number = %s", (citizenship_number,))
        if cur.fetchone():
            cur.close(); conn.close()
            return False, "Citizenship number already registered"

        if email:
            cur.execute("SELECT id FROM users WHERE email = %s", (email,))
            if cur.fetchone():
                cur.close(); conn.close()
                return False, "Email already registered"

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
        return True, "Registration successful"
    except Exception as e:
        print(f"Error registering user: {e}")
        return False, str(e)


# ── Violations ─────────────────────────────────────────────────────────────────

def save_violation(session_id, violation_type, vehicle_type,
                   timestamp_seconds, frame_number, confidence,
                   screenshot_path, video_filename,
                   license_plate=None, plate_confidence=None, plate_image_path=None):
    conn = get_db_connection()
    if not conn:
        return False
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO violations (
                session_id, violation_type, vehicle_type,
                timestamp_seconds, frame_number, confidence,
                screenshot_path, video_filename,
                license_plate, plate_confidence, plate_image_path
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            session_id, violation_type, vehicle_type,
            timestamp_seconds, frame_number, confidence,
            screenshot_path, video_filename,
            license_plate, plate_confidence, plate_image_path
        ))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving violation: {e}")
        conn.close()
        return False


def get_recent_sessions(limit=5):
    conn = get_db_connection()
    if not conn:
        return []
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT
                session_id,
                MAX(video_filename) as video_filename,
                COUNT(*) as violation_count,
                MAX(created_at) as last_processed
            FROM violations
            GROUP BY session_id
            ORDER BY last_processed DESC
            LIMIT %s
        """, (limit,))
        results = cur.fetchall()
        cur.close()
        conn.close()
        return results
    except Exception as e:
        print(f"Error fetching recent sessions: {e}")
        return []


# ── Admin: Stats ───────────────────────────────────────────────────────────────

def get_admin_stats():
    conn = get_db_connection()
    if not conn:
        return {}
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute("SELECT COUNT(*) as total FROM violations")
        total_violations = cur.fetchone()['total']

        cur.execute("SELECT COUNT(*) as total FROM violations WHERE violation_type = 'red_light'")
        red_light = cur.fetchone()['total']

        cur.execute("SELECT COUNT(*) as total FROM violations WHERE violation_type = 'speeding'")
        speeding = cur.fetchone()['total']

        cur.execute("SELECT COUNT(*) as total FROM violations WHERE violation_type = 'helmet'")
        helmet = cur.fetchone()['total']

        cur.execute("SELECT COUNT(*) as total FROM users")
        total_officers = cur.fetchone()['total']

        cur.execute("SELECT COUNT(*) as total FROM users WHERE is_active = TRUE")
        active_officers = cur.fetchone()['total']

        cur.execute("SELECT COUNT(*) as total FROM fines")
        total_fines = cur.fetchone()['total']

        cur.execute("SELECT COUNT(*) as total FROM fines WHERE status = 'unpaid'")
        unpaid_fines = cur.fetchone()['total']

        cur.execute("SELECT COALESCE(SUM(fine_amount), 0) as total FROM fines WHERE status = 'paid'")
        collected = cur.fetchone()['total']

        cur.execute("""
            SELECT DATE(created_at) as day, COUNT(*) as count
            FROM violations
            WHERE created_at >= NOW() - INTERVAL '7 days'
            GROUP BY DATE(created_at)
            ORDER BY day
        """)
        daily = cur.fetchall()

        cur.execute("""
            SELECT violation_type, COUNT(*) as count
            FROM violations
            GROUP BY violation_type
        """)
        by_type = cur.fetchall()

        cur.close()
        conn.close()

        return {
            'total_violations': int(total_violations),
            'red_light':        int(red_light),
            'speeding':         int(speeding),
            'helmet':           int(helmet),
            'total_officers':   int(total_officers),
            'active_officers':  int(active_officers),
            'total_fines':      int(total_fines),
            'unpaid_fines':     int(unpaid_fines),
            'collected_amount': float(collected),
            'daily_violations': [{'day': str(r['day']), 'count': int(r['count'])} for r in daily],
            'by_type':          [{'type': r['violation_type'], 'count': int(r['count'])} for r in by_type],
        }
    except Exception as e:
        print(f"Error fetching admin stats: {e}")
        return {
            'total_violations': 0, 'red_light': 0, 'speeding': 0, 'helmet': 0,
            'total_officers': 0, 'active_officers': 0, 'total_fines': 0,
            'unpaid_fines': 0, 'collected_amount': 0.0,
            'daily_violations': [], 'by_type': []
        }


# ── Admin: Officers ────────────────────────────────────────────────────────────

def get_all_officers():
    conn = get_db_connection()
    if not conn:
        return []
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT id, first_name, last_name, citizenship_number,
                   mobile_number, email, is_active, created_at, last_login
            FROM users
            ORDER BY created_at DESC
        """)
        results = cur.fetchall()
        cur.close()
        conn.close()
        return results
    except Exception as e:
        print(f"Error fetching officers: {e}")
        return []


def toggle_officer_status(user_id, is_active):
    conn = get_db_connection()
    if not conn:
        return False
    try:
        cur = conn.cursor()
        cur.execute("UPDATE users SET is_active = %s WHERE id = %s", (is_active, user_id))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error toggling officer status: {e}")
        return False


def delete_officer(user_id):
    conn = get_db_connection()
    if not conn:
        return False
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error deleting officer: {e}")
        return False


# ── Admin: Violations ──────────────────────────────────────────────────────────

def get_all_violations(limit=100, offset=0, vtype=None):
    conn = get_db_connection()
    if not conn:
        return []
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        if vtype:
            cur.execute("""
                SELECT v.*, f.id as fine_id, f.fine_amount, f.status as fine_status
                FROM violations v
                LEFT JOIN fines f ON f.violation_id = v.id
                WHERE v.violation_type = %s
                ORDER BY v.created_at DESC
                LIMIT %s OFFSET %s
            """, (vtype, limit, offset))
        else:
            cur.execute("""
                SELECT v.*, f.id as fine_id, f.fine_amount, f.status as fine_status
                FROM violations v
                LEFT JOIN fines f ON f.violation_id = v.id
                ORDER BY v.created_at DESC
                LIMIT %s OFFSET %s
            """, (limit, offset))
        results = cur.fetchall()
        cur.close()
        conn.close()
        return results
    except Exception as e:
        print(f"Error fetching violations: {e}")
        return []


# ── Admin: Fines ───────────────────────────────────────────────────────────────

def issue_fine(violation_id, session_id, license_plate, violation_type,
               fine_amount, issued_by, notes=None):
    conn = get_db_connection()
    if not conn:
        return False
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO fines (
                violation_id, session_id, license_plate,
                violation_type, fine_amount, issued_by, notes
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (violation_id, session_id, license_plate,
              violation_type, fine_amount, issued_by, notes))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error issuing fine: {e}")
        return False


def get_all_fines(limit=100, offset=0, status=None):
    conn = get_db_connection()
    if not conn:
        return []
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        if status:
            cur.execute("""
                SELECT f.*, u.first_name || ' ' || u.last_name as issued_by_name
                FROM fines f
                LEFT JOIN users u ON u.id = f.issued_by
                WHERE f.status = %s
                ORDER BY f.issued_at DESC
                LIMIT %s OFFSET %s
            """, (status, limit, offset))
        else:
            cur.execute("""
                SELECT f.*, u.first_name || ' ' || u.last_name as issued_by_name
                FROM fines f
                LEFT JOIN users u ON u.id = f.issued_by
                ORDER BY f.issued_at DESC
                LIMIT %s OFFSET %s
            """, (limit, offset))
        results = cur.fetchall()
        cur.close()
        conn.close()
        return results
    except Exception as e:
        print(f"Error fetching fines: {e}")
        return []


def update_fine_status(fine_id, status):
    conn = get_db_connection()
    if not conn:
        return False
    try:
        cur = conn.cursor()
        if status == 'paid':
            cur.execute(
                "UPDATE fines SET status = %s, paid_at = NOW() WHERE id = %s",
                (status, fine_id)
            )
        else:
            cur.execute(
                "UPDATE fines SET status = %s WHERE id = %s",
                (status, fine_id)
            )
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error updating fine status: {e}")
        return False


# ── Admin: News ────────────────────────────────────────────────────────────────

def get_all_news():
    conn = get_db_connection()
    if not conn:
        return []
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT n.*, a.name as author
            FROM news n
            LEFT JOIN admins a ON a.id = n.created_by
            ORDER BY n.created_at DESC
        """)
        results = cur.fetchall()
        cur.close()
        conn.close()
        return results
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []


def get_published_news():
    conn = get_db_connection()
    if not conn:
        return []
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT n.*, a.name as author
            FROM news n
            LEFT JOIN admins a ON a.id = n.created_by
            WHERE n.is_published = TRUE
            ORDER BY n.created_at DESC
            LIMIT 20
        """)
        results = cur.fetchall()
        cur.close()
        conn.close()
        return results
    except Exception as e:
        print(f"Error fetching published news: {e}")
        return []


def create_news(title, content, created_by):
    conn = get_db_connection()
    if not conn:
        return False
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO news (title, content, created_by)
            VALUES (%s, %s, %s)
        """, (title, content, created_by))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error creating news: {e}")
        return False


def update_news(news_id, title, content, is_published):
    conn = get_db_connection()
    if not conn:
        return False
    try:
        cur = conn.cursor()
        cur.execute("""
            UPDATE news SET title = %s, content = %s, is_published = %s
            WHERE id = %s
        """, (title, content, is_published, news_id))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error updating news: {e}")
        return False


def delete_news(news_id):
    conn = get_db_connection()
    if not conn:
        return False
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM news WHERE id = %s", (news_id,))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error deleting news: {e}")
        return False
