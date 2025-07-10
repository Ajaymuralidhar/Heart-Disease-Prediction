import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
device = torch.device("cpu")  # Explicitly use CPU

# --------- Preprocessing + Grad-CAM ----------
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor, image

def generate_gradcam(model, input_tensor, class_idx):
    model.eval()
    gradients, activations = [], []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    final_conv = model.features[-1]
    forward_handle = final_conv.register_forward_hook(forward_hook)
    backward_handle = final_conv.register_backward_hook(backward_hook)

    output = model(input_tensor)
    model.zero_grad()
    class_score = output[0, class_idx]
    class_score.backward()

    forward_handle.remove()
    backward_handle.remove()

    grads_val = gradients[0].cpu().data.numpy()[0]
    acts_val = activations[0].cpu().data.numpy()[0]
    weights = np.mean(grads_val, axis=(1, 2))

    cam = np.zeros(acts_val.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts_val[i]

    cam = np.maximum(cam, 0)
    cam = cam / cam.max() if cam.max() != 0 else cam
    return cam

def extract_bounding_box(cam):
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    cam_uint8 = np.uint8(255 * cam)
    thresh = cv2.threshold(cam_uint8, 150, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return x, y, w, h
    return None

def overlay_heatmap(img_path, cam, save_path='static/heatmaps'):
    img = cv2.imread(img_path)
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlaid = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    bbox = extract_bounding_box(cam)
    if bbox:
        x, y, w, h = bbox
        cv2.rectangle(overlaid, (x, y), (x + w, y + h), (0, 255, 0), 2)

    os.makedirs(save_path, exist_ok=True)
    out_path = os.path.join(save_path, os.path.basename(img_path))
    cv2.imwrite(out_path, overlaid)
    return out_path
