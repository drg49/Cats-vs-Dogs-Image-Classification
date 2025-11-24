# Cats vs Dogs Classifier 🐱🐶

A simple image classifier that distinguishes cats from dogs using deep learning. Built with **PyTorch** and trained on the [Microsoft Cats vs Dogs Dataset](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset).

In this project, the nueral network is built from scratch, to better understand how it works. In a future project, I will use transfer learning with ResNet.

## Features
- Binary classification: Cat or Dog  
- Uses **transfer learning** with pre-trained models  
- Data augmentation with horizontal flips and resizing   

## Dataset
The dataset contains thousands of labeled images of cats and dogs. Download it from [Kaggle](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset) and place it in a `data/` folder with the structure:


## Notes

ImageFolder follows a strict folder structure (which is the same folder structure we have in this project). 

Each subfolder under the `train/`, `val/`, & `test/` folders is treated as a class label. For example: `train/cats` would be 1 class label, and `train/dogs` would be the other class label (cats and dogs).

Images inside the subfolder are automatically assigned that label. In this case we have binary classification (cats = 0, dogs = 1)

- Example: **cats/cat1.jpg → label 0** (since cats is first alphabetically), then **dogs/dog1.jpg → label 1**

<br>Using `DataLoader`, cat & dog images are placed into batches of 32 for better performance.