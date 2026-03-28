"""
Database Configuration
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
    """Get PostgreSQL database connection"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

def save_violation(session_id, violation_type, vehicle_type, timestamp_seconds, 
                   frame_number, confidence, screenshot_path, video_filename,
                   license_plate=None, plate_confidence=None, plate_image_path=None):
    """
    Save violation to database with license plate data
    """
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cur = conn.cursor()
        
        cur.execute("""
            INSERT INTO violations (
                session_id, violation_type, vehicle_type,
                timestamp_seconds, frame_number, confidence,
                screenshot_path, video_filename, license_plate,
                plate_confidence, plate_image_path
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            session_id, 
            violation_type, 
            vehicle_type,
            timestamp_seconds, 
            frame_number, 
            confidence,
            screenshot_path, 
            video_filename, 
            license_plate,
            plate_confidence,
            plate_image_path
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
    """Get list of recent processing sessions with violation counts"""
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

def get_violations_by_session(session_id):
    """Get all violations for a specific session"""
    conn = get_db_connection()
    if not conn:
        return []
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT * FROM violations 
            WHERE session_id = %s 
            ORDER BY timestamp_seconds
        """, (session_id,))
        results = cur.fetchall()
        cur.close()
        conn.close()
        return results
    except Exception as e:
        print(f"Error fetching violations: {e}")
        return []