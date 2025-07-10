from rapidfuzz import process
faq_data = {
    ##General Heart Disease FAQs
    "what is heart disease": "Heart disease refers to various types of conditions that affect the heart's structure and function, such as coronary artery disease, arrhythmias, and congenital heart defects.",
    "what are the main types of heart disease": "The main types include coronary artery disease, heart failure, arrhythmia, heart valve problems, and congenital heart defects.",
    "what is myocardial infarction": "Myocardial Infarction, commonly known as a heart attack, occurs when blood flow to the heart muscle is blocked, causing tissue damage.",
    "what is abnormal heartbeat": "An abnormal heartbeat, or arrhythmia, is a condition where the heart beats too fast, too slow, or irregularly.",
    "what does it mean to have a history of myocardial infarction": "It means the patient has previously experienced a heart attack, which may increase the risk for future cardiovascular problems.",
    "what are the symptoms of a heart attack": "Common symptoms include chest pain, shortness of breath, sweating, nausea, and pain radiating to the arm or jaw.",
    "can heart disease be cured": "While not always curable, heart disease can often be managed with lifestyle changes, medication, and sometimes surgery.",
    ##ECG & Diagnosis
    "what is an ecg": "An ECG (electrocardiogram) records the electrical activity of the heart to help diagnose heart problems.",
    "how does an ecg detect heart disease": "ECGs detect abnormalities in heart rhythm, rate, and electrical conduction, helping to diagnose conditions like heart attacks or arrhythmias.",
    "can ecg detect heart attack": "Yes, an ECG can detect patterns that indicate a current or past heart attack.",
    "how accurate is ecg in diagnosing heart disease": "ECG is a reliable tool, especially when combined with other clinical assessments and imaging techniques.",
    "what are the limitations of ecg": "ECG may miss intermittent or early-stage abnormalities; further tests may be needed.",
    ##Risk Factors & Prevention
    "what are risk factors for heart disease": "Major risk factors include smoking, high blood pressure, diabetes, high cholesterol, obesity, and family history.",
    "how to prevent heart disease": "Preventive steps include a healthy diet, regular exercise, quitting smoking, and managing stress and blood pressure.",
    "does stress cause heart disease": "Chronic stress may contribute to heart disease by increasing blood pressure and promoting unhealthy behaviors.",
    ##Treatment & Management
    "how is heart disease treated": "Treatment can include lifestyle changes, medications, cardiac procedures, or surgery depending on the type and severity.",
    "can lifestyle changes reverse heart disease": "In some cases, especially early-stage, lifestyle changes can improve heart health significantly.",
    "what medications are used for heart disease": "Common drugs include beta-blockers, ACE inhibitors, statins, and blood thinners.",
    ##Technical / App-related Questions
    "what formats of ecg images are supported": "You can upload .jpg, .jpeg, or .png files.",
    "how long does it take to get results": "The system predicts results instantly once an ECG image is uploaded.",
    "how accurate is this model": "The model was trained on real ECG data and shows high accuracy for detecting four major conditions.",
    "is this tool a replacement for medical advice": "No, this tool is meant for preliminary assistance and should not replace professional medical consultation.",
    "what should i do after getting a prediction": "If you receive an abnormal result, consult a healthcare provider for further diagnosis and advice.",
    ##MI
    "what is myocardial infarction": "Myocardial infarction, also known as a heart attack, occurs when blood flow to a part of the heart is blocked.",
    "what causes a heart attack": "Heart attacks are most often caused by a blockage in one or more coronary arteries due to plaque buildup.",
    "what are symptoms of a heart attack": "Common symptoms include chest pain, shortness of breath, nausea, cold sweats, and discomfort in arms or jaw.",
    "how is myocardial infarction treated": "Treatment includes medications, angioplasty, or surgery like coronary artery bypass grafting.",
    "can a heart attack be prevented": "Yes, with healthy lifestyle habits, controlling blood pressure, cholesterol, avoiding smoking, and exercising regularly.",
    "how long is recovery after a heart attack": "Recovery varies, but cardiac rehab usually lasts 3 to 6 months.",
    "is heart attack the same as cardiac arrest": "No, a heart attack is a blood flow problem; cardiac arrest is an electrical problem causing the heart to stop.",
    ##AHB
    "what is an abnormal heartbeat": "An abnormal heartbeat, or arrhythmia, is when the heart beats too fast, too slow, or irregularly.",
    "is arrhythmia dangerous": "Some arrhythmias are harmless, while others can lead to serious complications like stroke or heart failure.",
    "how is arrhythmia diagnosed": "It is diagnosed using ECG, Holter monitoring, or electrophysiology studies.",
    "what causes irregular heartbeats": "Causes include heart disease, stress, caffeine, alcohol, or electrolyte imbalance.",
    "can arrhythmias be cured": "Many arrhythmias are treatable with medication, ablation therapy, or pacemakers.",
    "should I worry about palpitations": "Occasional palpitations are common, but frequent or prolonged symptoms should be evaluated by a doctor.",
    ##PMI
    "what should i do after a heart attack": "Follow your doctor's advice, attend cardiac rehab, take medications, and make lifestyle changes.",
    "can i exercise after a heart attack": "Yes, but under medical supervision. Cardiac rehab includes safe exercise routines.",
    "how to prevent another heart attack": "Take medications regularly, quit smoking, eat healthy, and manage stress.",
    "is life normal after a heart attack": "Many people return to normal life with some adjustments in diet, exercise, and stress levels.",
    "do i need lifelong medication after MI": "Often yes, medications like beta-blockers, statins, and antiplatelets are prescribed long term.",
    "how often should i follow up after MI": "Initial follow-ups are frequent. Over time, check-ups are usually every 3–6 months depending on condition.",
    ##NHB
    "how to keep heart healthy": "Eat a balanced diet, exercise regularly, avoid smoking, manage stress, and get regular check-ups.",
    "what is a normal heart rate": "For adults, a normal resting heart rate is typically between 60 to 100 beats per minute.",
    "how often should i check my heart": "Yearly checkups are recommended, especially if you're over 40 or have risk factors.",
    "are ECG tests safe": "Yes, ECGs are non-invasive, painless, and completely safe diagnostic tools.",
    "can heart disease be genetic": "Yes, genetics can increase your risk, but lifestyle factors also play a big role.",
    "do i need an ECG if i feel healthy": "Not always, but it may be done if there's a family history or symptoms like chest pain or palpitations.",
    ##EXTRAS
    "what is coronary artery disease": "Coronary artery disease is the narrowing or blockage of the coronary arteries due to plaque buildup.",
    "what is angina": "Angina is chest pain or discomfort caused by reduced blood flow to the heart.",
    "what causes heart failure": "Heart failure can be caused by coronary artery disease, high blood pressure, or previous heart attacks.",
    "can valve disease cause symptoms": "Yes, symptoms may include fatigue, shortness of breath, chest pain, and swelling in the ankles.",
    "what happens during a heart attack": "A coronary artery gets blocked, preventing oxygen-rich blood from reaching part of the heart muscle.",
    "what tests confirm a heart attack": "ECG, blood tests (troponin), and coronary angiography help confirm a heart attack.",
    "are silent heart attacks dangerous": "Yes, they cause heart muscle damage even without symptoms and can lead to complications.",
    "what is atrial fibrillation": "Atrial fibrillation is a common type of arrhythmia where the heart beats irregularly and often rapidly.",
    "can caffeine trigger irregular heartbeat": "In some individuals, caffeine can increase the risk of palpitations or arrhythmias.",
    "are all arrhythmias life-threatening": "No, some are harmless while others may require urgent treatment.",
    "can ecg detect heart enlargement": "Yes, certain patterns in ECG can suggest hypertrophy or enlargement of heart chambers.",
    "why do i need an ecg if i feel fine": "ECG can detect silent or early changes that may not yet cause symptoms.",
    "how often should ecg be done": "It depends on your risk profile, age, and symptoms—consult your physician.",
    "does high cholesterol cause heart disease": "Yes, it contributes to plaque formation in arteries leading to heart attacks.",
    "is diabetes a heart disease risk factor": "Yes, people with diabetes have a higher risk of cardiovascular diseases.",
    "can quitting smoking reverse heart damage": "It significantly reduces future risk, but some damage may be irreversible.",
    "what is cardiac rehabilitation": "It's a supervised program to help patients recover after a heart attack or surgery.",
    "can i drive after a heart attack": "Typically, after a few weeks of recovery, but check with your doctor.",
    "what is a heart-healthy diet": "A diet rich in fruits, vegetables, lean proteins, whole grains, and low in salt and saturated fats.",
    "why are blood thinners prescribed after mi": "To prevent blood clots and reduce the risk of another heart attack.",
    "what are statins used for": "Statins help lower LDL cholesterol and reduce heart disease risk.",
    "do i need to monitor my blood pressure daily": "Yes, if you have hypertension or are recovering from a cardiac event.",
    "can i use this tool at home for regular checks": "Yes, but it's not a substitute for clinical testing or diagnosis.",
    "does this tool work on mobile devices": "Yes, as long as you can upload ECG image files.",
    "how secure is my uploaded data": "Uploaded images are processed locally and not stored permanently."

}


from rapidfuzz import process

def get_faq_response(user_input):
    user_input = user_input.lower()
    question_match, score, _ = process.extractOne(user_input, faq_data.keys())

    if score > 70:
        return faq_data[question_match]
    else:
        return "Sorry, I couldn't understand your question. Try asking about ECG, MI, or heart attack."
