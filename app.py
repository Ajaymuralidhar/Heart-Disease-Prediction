import os
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename

from utils.model_loader import load_model, predict
from faq_bot import get_faq_response
from utils.gradcam import preprocess_image, generate_gradcam, overlay_heatmap
from utils.database import init_db, insert_patient, get_all_patients
from utils.auth import check_login, login_user, is_logged_in, logout_user

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

            # Save to DB
            insert_patient(name, age, gender, symptoms, result, gradcam_path)

            # Save in session for report and chatbot
            session['last_analysis'] = {
                'name': name,
                'age': age,
                'gender': gender,
                'symptoms': symptoms,
                'result': result,
                'image_path': save_path,
                'gradcam_path': gradcam_path,
                'confidence_scores': confidence_scores
            }

            return redirect('/report')

        except Exception as e:
            return f"Error during analysis: {e}", 500

    return render_template('analyze.html')

@app.route('/report')
def report():
    if not is_logged_in():
        return redirect('/login')

    context = session.get('last_analysis')
    if not context:
        return redirect('/analyze')

    return render_template("report.html", **context)

@app.route('/faq', methods=['GET', 'POST'])
def faq():
    if request.method == 'POST':
        user_question = request.form['question']
        answer = get_faq_response(user_question)

        context = session.get('last_analysis', {})
        return render_template('chatbot.html', chatbot_response=answer, **context)

    return render_template('chatbot.html', **session.get('last_analysis', {}))

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
