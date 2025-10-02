import sqlite3

DB_NAME = "patients.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    age INTEGER,
                    gender TEXT,
                    symptoms TEXT,
                    result TEXT,
                    gradcam_path TEXT,
                    gemini_explanation TEXT
                )''')
    conn.commit()
    conn.close()

def insert_patient(name, age, gender, symptoms, result, gradcam_path, gemini_explanation):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''INSERT INTO patients (name, age, gender, symptoms, result, gradcam_path, gemini_explanation)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''',
              (name, age, gender, symptoms, result, gradcam_path, gemini_explanation))
    conn.commit()
    conn.close()

def get_patient_by_id(patient_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT * FROM patients WHERE id=?", (patient_id,))
    row = c.fetchone()
    conn.close()
    return row

def get_all_patients():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT * FROM patients")
    rows = c.fetchall()
    conn.close()
    return rows

def delete_patient(patient_id):
    import sqlite3
    conn = sqlite3.connect('patients.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
    conn.commit()
    conn.close()