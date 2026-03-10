# db.py
import psycopg2

def get_connection():
    """
    Returns a connection to the TrafficDB PostgreSQL database.
    Make sure your DB, user, and password are correct.
    """
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="TrafficDB",
            user="postgres",
            password="postgres",  # <-- check your actual postgres password
            port="5432"
        )
        return conn
    except Exception as e:
        print("ERROR: Could not connect to database:", e)
        return None


def insert_citizen(first_name, last_name, citizenship_number, issued_district,
                   date_of_birth, mobile_number, email, password):
    """
    Inserts a new citizen record into the citizens table.
    """
    conn = get_connection()
    if conn is None:
        print("Insert failed: no DB connection")
        return False

    try:
        cursor = conn.cursor()
        query = """
            INSERT INTO citizens 
            (first_name, last_name, citizenship_number, issued_district, date_of_birth, mobile_number, email, password)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (
            first_name, last_name, citizenship_number, issued_district,
            date_of_birth, mobile_number, email, password
        ))
        conn.commit()
        cursor.close()
        conn.close()
        print("Citizen inserted successfully ✅")
        return True

    except Exception as e:
        print("Database error:", e)
        return False


# Optional: test run
if __name__ == "__main__":
    # Hardcoded test insert
    success = insert_citizen(
        "Sabin", "Adhikari", "12-34-567890", "Kathmandu",
        "2003-05-21", "+9779812345678", "sabin@example.com", "Test@1234"
    )
    if success:
        print("Test citizen added to DB")