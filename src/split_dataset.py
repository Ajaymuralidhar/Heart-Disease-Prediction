import os
import shutil
import random
from tqdm import tqdm

SOURCE_DIR = 'data/processed'
DEST_DIR = 'data/split'
SPLIT_RATIO = 0.8  # 80% train, 20% val

def create_split_dirs():
    for split in ['train', 'val']:
        split_path = os.path.join(DEST_DIR, split)
        if not os.path.exists(split_path):
            os.makedirs(split_path)
        for class_name in os.listdir(SOURCE_DIR):
            class_path = os.path.join(split_path, class_name)
            os.makedirs(class_path, exist_ok=True)

def split_data():
    create_split_dirs()

    for class_name in os.listdir(SOURCE_DIR):
        class_dir = os.path.join(SOURCE_DIR, class_name)
        images = os.listdir(class_dir)
        random.shuffle(images)

        split_idx = int(SPLIT_RATIO * len(images))
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        # Copy images
        for img in tqdm(train_images, desc=f"Train - {class_name}"):
            src = os.path.join(class_dir, img)
            dst = os.path.join(DEST_DIR, 'train', class_name, img)
            shutil.copyfile(src, dst)

        for img in tqdm(val_images, desc=f"Val - {class_name}"):
            src = os.path.join(class_dir, img)
            dst = os.path.join(DEST_DIR, 'val', class_name, img)
            shutil.copyfile(src, dst)

if __name__ == "__main__":
    split_data()
