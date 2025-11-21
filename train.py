import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# -------------------------
# 1. DATA TRANSFORMS
# -------------------------
# Transform each image in the training folder
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# Transform each image in the validation folder
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Transform each image in the test folder (same as validation)
test_transforms = val_transforms

# -------------------------
# 2. LOAD DATASETS
# -------------------------
train_dataset = datasets.ImageFolder("data/train", transform=train_transforms)
val_dataset   = datasets.ImageFolder("data/val",   transform=val_transforms)
test_dataset  = datasets.ImageFolder("data/test",  transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32)
test_loader  = DataLoader(test_dataset, batch_size=32)

print("Classes:", train_dataset.classes)
