import os
import re
import json
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai

# ---------------------------------------------------------
# Environment & Gemini configuration
# ---------------------------------------------------------

# Load variables from .env (override any OS-level env vars)
load_dotenv(override=True)

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is not set. Check your .env file.")

genai.configure(api_key=API_KEY)

# Load Gemini model
model = genai.GenerativeModel("gemini-2.5-flash")


# ---------------------------------------------------------
# ECG + Grad-CAM analysis helper
# ---------------------------------------------------------

def analyze_with_gemini(gradcam_path: str, symptoms: str) -> dict:
    """
    Analyze an ECG Grad-CAM image plus patient symptoms using Gemini.

    Returns a dict with this exact structure (safe for Flask templates):
        {
            "observation": str,
            "risks": [str, ...],
            "meaning": str,
            "next_steps": [str, ...]
        }
    """
    try:
        # Load Grad-CAM image
        gradcam_img = Image.open(gradcam_path)

        # Prompt Gemini: image + system-style instructions + symptoms
        response = model.generate_content([
            gradcam_img,
            f"""
            You are a medical assistant with expertise in ECG interpretation.
            Use the provided Grad-CAM image and the patient's symptoms
            to generate a structured clinical analysis.

            IMPORTANT INSTRUCTIONS:
            - Output MUST be ONLY valid JSON (no markdown, no text outside JSON).
            - Do NOT wrap JSON in backticks or code fences.
            - Use the exact keys listed below.
            - Provide detailed, accurate responses based on standard ECG interpretation guidelines and the Grad-CAM visualization.
            - Base your analysis on evidence from the image and symptoms; avoid speculation.
            - If uncertain, clearly express uncertainty instead of hallucinating.
            - Ensure responses are clinically relevant and comprehensive without being overly verbose.

            REQUIRED JSON FORMAT:
            {{
                "observation": "<detailed clinical observation from Grad-CAM, describing key features and patterns>",
                "risks": ["<potential risk 1>", "<potential risk 2>", "<additional risks if applicable>"],
                "meaning": "<detailed explanation of what the findings could indicate, including possible conditions>",
                "next_steps": ["<recommended action 1>", "<recommended action 2>", "<additional steps if needed>"]
            }}

            Patient symptoms: {symptoms}
            """
        ])

        # Get text response
        text = (response.text or "").strip()

        # In case the model still adds code fences like ```json ... ```
        text = re.sub(r'```json|```', '', text).strip()

        # Extract the first JSON object from the response
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            text = match.group(0)

        # Parse JSON
        data = json.loads(text)

        # Ensure full structure with sane defaults
        result = {
            "observation": data.get("observation", "").strip(),
            "risks": data.get("risks", []) or [],
            "meaning": data.get("meaning", "").strip(),
            "next_steps": data.get("next_steps", []) or []
        }

        # Force list types for robustness
        if not isinstance(result["risks"], list):
            result["risks"] = [str(result["risks"])]
        if not isinstance(result["next_steps"], list):
            result["next_steps"] = [str(result["next_steps"])]

        return result

    except Exception as e:
        # Log the error server-side for debugging
        print("Gemini analysis error:", repr(e))

        # Safe, user-facing fallback
        return {
            "observation": f"⚠️ Gemini error: {e}",
            "risks": [],
            "meaning": "Unable to analyze ECG automatically due to an API or parsing error.",
            "next_steps": [
                "Verify the API key and network connection.",
                "Try again after a short while.",
                "Seek review by a qualified clinician for final interpretation."
            ]
        }
