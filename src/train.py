import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Config
TRAIN_DIR = 'data/split/train'
VAL_DIR = 'data/split/val'
MODEL_SAVE_PATH = 'saved_models/ecg_model.pth'
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001
NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image Transforms
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # MobileNetV2 expects 3-channel input
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# Load Datasets
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
class_names = train_dataset.classes

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Compute class weights from training data
train_labels = [label for _, label in train_dataset]
class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(train_labels),
                                     y=train_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

# Model setup
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
model = model.to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training function
def train():
    print("🔧 Starting training...\n")
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels).item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total
        train_loss /= train_total

        # Validation loop
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels).item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        val_loss /= val_total

        print(f"📍 Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f} - Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} - Acc: {val_acc:.4f}")

        # Save model if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("✅ Best model saved!\n")

    print(f"\n🎉 Training complete! Best Val Accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    train()



