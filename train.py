import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# -------------------------------------------
# 1. TRANSFORMS (resize, augment, to tensor)
# -------------------------------------------
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_transforms = val_transforms

# -------------------------------------------
# 2. LOAD DATASETS FROM FOLDERS
# -------------------------------------------
train_dataset = datasets.ImageFolder("data/train", transform=train_transforms)
val_dataset   = datasets.ImageFolder("data/val",   transform=val_transforms)
test_dataset  = datasets.ImageFolder("data/test",  transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32)
test_loader  = DataLoader(test_dataset, batch_size=32)

print("Classes:", train_dataset.classes)  # Expect: ['Cat', 'Dog']

# -------------------------------------------
# 3. DEFINE CNN MODEL
# -------------------------------------------
# Inherit from nn.Module to create a custom CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Feature extractor (Convolutional layers)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Classifier (Fully connected layers)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # 2 classes: Cat/Dog
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# Instantiate the model
model = SimpleCNN()

# -------------------------------------------
# 4. TRAINING SETUP (loss, optimizer)
# -------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------------------------
# 5. TRAINING LOOP
# -------------------------------------------
epochs = 5

for epoch in range(epochs):
    model.train()
    train_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

# -------------------------------------------
# 6. VALIDATION ACCURACY
# -------------------------------------------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

val_acc = correct / total
print(f"Validation Accuracy: {val_acc:.2f}")

# -------------------------------------------
# 7. SAVE THE MODEL
# -------------------------------------------
torch.save(model.state_dict(), "cats_dogs_cnn.pth")
print("Model saved as cats_dogs_cnn.pth")
