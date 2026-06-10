import sqlite3


def init_db():
    conn = sqlite3.connect("patients.db")
    cursor = conn.cursor()

    # Users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )
    """)

    # Predictions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        age REAL,
        sex REAL,
        cp REAL,
        trestbps REAL,
        chol REAL,
        fbs REAL,
        restecg REAL,
        thalach REAL,
        exang REAL,
        oldpeak REAL,
        slope REAL,
        ca REAL,
        thal REAL,
        result TEXT,
        confidence REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


def fix_database():
    conn = sqlite3.connect("patients.db")
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(predictions)")
    existing = [col[1] for col in cursor.fetchall()]

    required_columns = {
        "username": "TEXT",
        "age": "REAL",
        "sex": "REAL",
        "cp": "REAL",
        "trestbps": "REAL",
        "chol": "REAL",
        "fbs": "REAL",
        "restecg": "REAL",
        "thalach": "REAL",
        "exang": "REAL",
        "oldpeak": "REAL",
        "slope": "REAL",
        "ca": "REAL",
        "thal": "REAL",
        "result": "TEXT",
        "confidence": "REAL",
        "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    }

    for col, col_type in required_columns.items():
        if col not in existing:
            cursor.execute(
                f"ALTER TABLE predictions ADD COLUMN {col} {col_type}"
            )

    conn.commit()
    conn.close()