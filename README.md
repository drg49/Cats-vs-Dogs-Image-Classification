# Cats vs Dogs Classifier 🐱🐶

Google Colab Project for this notebook: <br>
https://colab.research.google.com/drive/1Qe4fry0cijn-4QXljEVqrz4WpgOt_Yws#scrollTo=Mn8mOf8v93Qo

<br>

A simple image classifier that distinguishes cats from dogs using deep learning. Built with **PyTorch** and trained on the [Microsoft Cats vs Dogs Dataset](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset).

In this project, the nueral network is built from scratch, to better understand how it works. In a future project, I will use transfer learning with ResNet.

## Features

- Binary classification: Cat or Dog
- Uses **transfer learning** with pre-trained models
- Data augmentation with horizontal flips and resizing

## Dataset

The dataset contains thousands of labeled images of cats and dogs. Download it from [Kaggle](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset) and place it in the project. Create a `split_dataset_move.py` in the same directory to organize images in the correct structure. Here is the script:

```
import os
import shutil
import random
from pathlib import Path
from PIL import Image

# ---- Config ----
SOURCE_DIR = "PetImages"
TARGET_DIR = "data"
CATEGORIES = ["Cat", "Dog"]

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1   # test split = remaining 0.1

# ---- Create destination folders ----
print("Creating dataset folders...")
for split in ["train", "val", "test"]:
    for cls in ["cats", "dogs"]:
        out_dir = Path(TARGET_DIR) / split / cls
        out_dir.mkdir(parents=True, exist_ok=True)

def is_image_corrupted(path):
    """Try opening the image; return True if corrupted."""
    try:
        img = Image.open(path)
        img.verify()  # validate file
        return False
    except Exception:
        return True

# ---- Process Cat & Dog folders ----
for category in CATEGORIES:
    print(f"\nProcessing: {category}")

    input_dir = Path(SOURCE_DIR) / category
    images = list(input_dir.glob("*"))

    clean_images = []
    for img_path in images:
        # Skip non-files
        if not img_path.is_file():
            continue

        if is_image_corrupted(img_path):
            print(f"  Corrupt image removed: {img_path}")
            img_path.unlink(missing_ok=True)  # delete corrupted file
        else:
            clean_images.append(img_path)

    # Shuffle images randomly
    random.shuffle(clean_images)

    total = len(clean_images)
    train_end = int(total * TRAIN_SPLIT)
    val_end = int(total * (TRAIN_SPLIT + VAL_SPLIT))

    train_imgs = clean_images[:train_end]
    val_imgs = clean_images[train_end:val_end]
    test_imgs = clean_images[val_end:]

    # Convert "Cat" → "cats", "Dog" → "dogs"
    cls_name = "cats" if category.lower() == "cat" else "dogs"

    print(f"  Train: {len(train_imgs)}")
    print(f"  Val:   {len(val_imgs)}")
    print(f"  Test:  {len(test_imgs)}")

    # ---- Move files into new structure ----
    for img_path in train_imgs:
        shutil.move(str(img_path), Path(TARGET_DIR) / "train" / cls_name / img_path.name)

    for img_path in val_imgs:
        shutil.move(str(img_path), Path(TARGET_DIR) / "val" / cls_name / img_path.name)

    for img_path in test_imgs:
        shutil.move(str(img_path), Path(TARGET_DIR) / "test" / cls_name / img_path.name)

print("\nDone! Images moved into the 'data/' folder.")
```

## Notes

ImageFolder follows a strict folder structure (which is the same folder structure we have in this project).

Each subfolder under the `train/`, `val/`, & `test/` folders is treated as a class label. For example: `train/cats` would be 1 class label, and `train/dogs` would be the other class label (cats and dogs).

Images inside the subfolder are automatically assigned that label. In this case we have binary classification (cats = 0, dogs = 1)

- Example: **cats/cat1.jpg → label 0** (since cats is first alphabetically), then **dogs/dog1.jpg → label 1**

<br>Using `DataLoader`, cat & dog images are placed into batches of **32** for better performance. GPUs often handle powers of 2 efficiently (16, 32, 64, 128…), 32 is a good choice here.
