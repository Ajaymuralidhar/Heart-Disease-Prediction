import sqlite3

def init_db():
    conn = sqlite3.connect('database/patients.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY,
            name TEXT,
            age INTEGER,
            gender TEXT,
            symptoms TEXT,
            prediction TEXT,
            gradcam_path TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def insert_patient(name, age, gender, symptoms, prediction, gradcam_path):
    conn = sqlite3.connect('database/patients.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO patients (name, age, gender, symptoms, prediction, gradcam_path)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (name, age, gender, symptoms, prediction, gradcam_path))
    conn.commit()
    conn.close()

def get_all_patients():
    conn = sqlite3.connect('database/patients.db')
    c = conn.cursor()
    c.execute("SELECT * FROM patients ORDER BY timestamp DESC")
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