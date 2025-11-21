# This file is used for visualizing and testing code snippets.
from PIL import Image
from torchvision import transforms


# -------------------------
# Visualize an Image as a Tensor
# -------------------------

# Load an image
img = Image.open(r"data\test\cats\5.jpg")

# Transform the image into a tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

img_tensor = transform(img)

# Should be [3, 224, 224]
# 3 = RGB channels
# 224x224 = height × width
print(f"Image tensor shape: {img_tensor.shape}")

print(img_tensor)