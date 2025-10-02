import os
import cv2
from tqdm import tqdm

# Configuration
INPUT_DIR = "data/raw"
OUTPUT_DIR = "data/processed"
IMAGE_SIZE = (224, 224)  # For MobileNetV2

def denoise_image(img, method="clahe"):
    """Apply denoising to the grayscale image."""
    if method == "gaussian":
        return cv2.GaussianBlur(img, (5, 5), 0)
    elif method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)
    else:
        return img

def preprocess_and_save_image(input_path, output_path, denoise_method="clahe"):
    """Preprocess a single image and save it."""
    try:
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Skipping unreadable file: {input_path}")
            return

        img = denoise_image(img, method=denoise_method)
        img = cv2.resize(img, IMAGE_SIZE)
        cv2.imwrite(output_path, img)
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def process_dataset(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, denoise_method="clahe"):
    """Process the entire dataset directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for class_name in os.listdir(input_dir):
        class_input_path = os.path.join(input_dir, class_name)
        class_output_path = os.path.join(output_dir, class_name)

        if not os.path.isdir(class_input_path):
            continue

        os.makedirs(class_output_path, exist_ok=True)

        print(f"Processing class: {class_name}")
        for filename in tqdm(os.listdir(class_input_path)):
            input_path = os.path.join(class_input_path, filename)
            output_path = os.path.join(class_output_path, filename)
            preprocess_and_save_image(input_path, output_path, denoise_method=denoise_method)

if __name__ == "__main__":
    process_dataset()
