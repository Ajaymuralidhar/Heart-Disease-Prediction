from flask import session
import firebase_admin
from firebase_admin import credentials, auth
import os

# --- Firebase Setup ---
# TODO: REPLACE 'firebase_key.json' with the path to your downloaded Firebase Service Account Key JSON file.
CRED_PATH = os.getenv("FIREBASE_CRED_PATH", "D:\projects\Heart_Disease_Detection\heart-disease-91cbd-firebase-adminsdk-fbsvc-1fc6e3f4bf.json")

# Initialize the Firebase Admin SDK once globally
try:
    if not firebase_admin._apps:
        cred = credentials.Certificate(CRED_PATH)
        firebase_admin.initialize_app(cred)
        print("Firebase Admin SDK initialized successfully.")
except Exception as e:
    # This will catch FileNotFoundError if the key is missing or other initialization errors.
    print(f"ERROR: Failed to initialize Firebase Admin SDK: {e}. Authentication will fail.")

# --- Authentication Logic (repurposing existing function names) ---

def check_login(id_token):
    """
    Verifies the Firebase ID token sent from the client. (Replaces old username/password check)
    
    Returns:
        dict: The decoded token (user data) if valid, or None if verification fails.
    """
    if not id_token:
        return None
    try:
        # Verify token's signature, expiry, and issuer
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token
    except Exception as e:
        print(f"Token verification failed: {e}")
        return None

def login_user(user_data):
    """Sets the Flask session upon successful token verification."""
    # Store minimal, secure info (like UID) in the session
    session['user'] = user_data['uid']
    session['user_email'] = user_data.get('email', 'N/A')

def is_logged_in():
    """Checks if a user UID exists in the session."""
    return 'user' in session

def logout_user():
    """Removes user info from the session."""
    session.pop('user', None)
    session.pop('user_email', None)