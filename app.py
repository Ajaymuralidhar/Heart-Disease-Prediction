import os
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename

from utils.model_loader import load_model, predict
from faq_bot import get_faq_response
from utils.gradcam import preprocess_image, generate_gradcam, overlay_heatmap
from utils.database import init_db, insert_patient, get_all_patients, get_patient_by_id
from utils.auth import check_login, login_user, is_logged_in, logout_user
from gemini import analyze_with_gemini   # <-- Gemini integration

# Initialize App
app = Flask(__name__)
app.secret_key = 'Doctor'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model and initialize DB
model = load_model()
init_db()
class_names = ['Abnormal', 'Myocardial Infarction', 'Normal', 'Patient MI History']

@app.route('/')
def home():
    return redirect('/login')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if check_login(request.form['username'], request.form['password']):
            login_user(request.form['username'])
            return redirect('/dashboard')
        else:
            return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if not is_logged_in():
        return redirect('/login')
    return render_template('dashboard.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if not is_logged_in():
        return redirect('/login')

    if request.method == 'POST':
        try:
            name = request.form['name']
            age = request.form['age']
            gender = request.form['gender']
            symptoms = request.form['symptoms']
            file = request.files['image']

            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)

            # Prediction
            pred_idx, probs = predict(model, save_path)
            result = class_names[pred_idx]
            confidence_scores = [(class_names[i], round(probs[i].item(), 2)) for i in range(len(class_names))]

            # Grad-CAM
            image_tensor, _ = preprocess_image(save_path)
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)
            cam = generate_gradcam(model, image_tensor, pred_idx)
            gradcam_path = overlay_heatmap(save_path, cam)

            # Gemini AI Explanation
            gemini_explanation = analyze_with_gemini(gradcam_path, symptoms)

            # Save to DB
            insert_patient(name, age, gender, symptoms, result, gradcam_path, gemini_explanation)

            # Save summary in session (for redirect)
            session['last_patient_name'] = name  

            return redirect('/report')

        except Exception as e:
            return f"Error during analysis: {e}", 500

    return render_template('analyze.html')

import json

@app.route('/report')
def report():
    if not is_logged_in():
        return redirect('/login')

    # Fetch last inserted patient from DB
    last_patient_name = session.get('last_patient_name')
    if not last_patient_name:
        return redirect('/analyze')

    patients = get_all_patients()
    patient = [p for p in patients if p[1] == last_patient_name][-1]  # latest entry

    # Try parsing Gemini explanation (stored as JSON string in DB)
    try:
        gemini_data = json.loads(patient[7]) if patient[7] else {}
    except:
        gemini_data = {"observation": patient[7], "risks": [], "meaning": "", "next_steps": []}

    context = {
        "name": patient[1],
        "age": patient[2],
        "gender": patient[3],
        "symptoms": patient[4],
        "result": patient[5],
        "gradcam_path": patient[6],
        "image_path": "static/uploads/" + os.path.basename(patient[6]).replace("_gradcam",""),
        "confidence_scores": [],  # could be extended if stored

        # Structured Gemini fields
        "gemini_observation": gemini_data.get("observation", ""),
        "gemini_risks": gemini_data.get("risks", []),
        "gemini_meaning": gemini_data.get("meaning", ""),
        "gemini_next_steps": gemini_data.get("next_steps", [])
    }

    return render_template("report.html", **context)


@app.route('/faq', methods=['GET', 'POST'])
def faq():
    if request.method == 'POST':
        user_question = request.form['question']
        answer = get_faq_response(user_question)

        return render_template('chatbot.html', chatbot_response=answer)

    return render_template('chatbot.html')

@app.route('/history')
def history():
    if not is_logged_in():
        return redirect('/login')
    patients = get_all_patients()
    return render_template('history.html', patients=patients)

@app.route('/logout')
def logout():
    logout_user()
    return redirect('/login')

@app.route('/delete/<int:patient_id>', methods=['POST'])
def delete_record(patient_id):
    if not is_logged_in():
        return redirect('/login')
    
    from utils.database import delete_patient
    delete_patient(patient_id)
    return redirect('/history')

if __name__ == '__main__':
    app.run(debug=True)
