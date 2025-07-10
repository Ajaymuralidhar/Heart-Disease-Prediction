import torch
from torchvision import models, transforms
from torchvision.models import mobilenet_v2
from PIL import Image
import cv2
import numpy as np

MODEL_PATH = "saved_models/ecg_model.pth"
NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
def load_model():
    model = mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# Preprocess the image for both prediction and Grad-CAM
def preprocess_image(image_path):
    # Load the image using OpenCV and convert to grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Invalid image data")

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Resize the image to 224x224
    img = cv2.resize(img, (224, 224))

    # Convert to RGB PIL image for transforms
    pil_img = Image.fromarray(img).convert("RGB")

    # Apply normalization transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    tensor = transform(pil_img).unsqueeze(0)  # Shape: [1, 3, 224, 224]
    return tensor, pil_img

# Make prediction
def predict(model, image_path):
    input_tensor, _ = preprocess_image(image_path)
    input_tensor = input_tensor.to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0).cpu().numpy()

    predicted_idx = probabilities.argmax()
    return predicted_idx, probabilities
