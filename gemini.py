import google.generativeai as genai
from PIL import Image
import json

# Configure Gemini API
GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY"
genai.configure(api_key=GOOGLE_API_KEY)

# Load Gemini model
model = genai.GenerativeModel("gemini-2.5-flash")

def analyze_with_gemini(gradcam_path: str, symptoms: str):
    """
    Send Grad-CAM + symptoms to Gemini and return structured explanation.
    """
    try:
        gradcam_img = Image.open(gradcam_path)

        # Ask Gemini to return JSON
        response = model.generate_content([
            gradcam_img,
            f"""
            You are a medical assistant specializing in heart conditions.

            Analyze the ECG Grad-CAM and the patient's symptoms.

            Return your answer strictly in JSON format with these keys:
            - observation: one short paragraph
            - risks: a list of possible heart risks or conditions
            - meaning: one short paragraph in patient-friendly language
            - next_steps: a list of clear recommended actions

            Symptoms: {symptoms}
            """
        ])

        # Parse Gemini JSON response
        try:
            data = json.loads(response.text)
            observation = data.get("observation", "")
            risks = data.get("risks", [])
            meaning = data.get("meaning", "")
            next_steps = data.get("next_steps", [])
        except:
            # Fallback if JSON parsing fails
            observation, risks, meaning, next_steps = response.text, [], "", []

        return {
            "observation": observation,
            "risks": risks,
            "meaning": meaning,
            "next_steps": next_steps
        }

    except Exception as e:
        return {
            "observation": "",
            "risks": [],
            "meaning": "",
            "next_steps": [f"⚠️ Error generating explanation: {e}"]
        }
