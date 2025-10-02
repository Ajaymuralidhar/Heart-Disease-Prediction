import torch
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import os

# Config
MODEL_PATH = 'saved_models/ecg_model.pth'
IMAGE_PATH = r'data\raw\Normal_Person\Normal(25).jpg'  # change this
CLASS_NAMES = ['Abnormal', 'Myocardial Infarction', 'Normal', 'History of MI']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing (must match training)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def preprocess_image(image_path):
    # Read the image as bytes
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    # Convert bytes to NumPy array for OpenCV
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Invalid image data")

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Resize to match model input
    img = cv2.resize(img, (224, 224))

    # Convert to RGB PIL image (needed for torchvision transforms)
    pil_img = Image.fromarray(img).convert("RGB")

    # Apply model transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(pil_img).unsqueeze(0)

# Load model
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.last_channel, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Load and preprocess image
image = preprocess_image(IMAGE_PATH).to(DEVICE)

# Predict
with torch.no_grad():
    outputs = model(image)
    _, pred = torch.max(outputs, 1)
    predicted_class = CLASS_NAMES[pred.item()]

print(f" Predicted Class: {predicted_class}")
