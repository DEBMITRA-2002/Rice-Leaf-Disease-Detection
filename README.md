
# 🌾 Rice Leaf Disease Detection using Deep Learning

An image classification project that uses Convolutional Neural Networks (CNNs) and MobileNetV2 transfer learning to classify rice leaf images into six dataset categories.

## Overview

Rice leaf diseases can affect crop health and agricultural productivity. This project explores deep learning-based image classification to identify visual patterns associated with different rice leaf categories.

The application provides a Streamlit interface for image upload, model prediction, class scores, and Grad-CAM visualization.

## Disease Categories

- Bacterial Leaf Blight
- Brown Spot
- Healthy Rice Leaf
- Leaf Blast
- Leaf Scald
- Sheath Blight

## Tech Stack

- Python
- TensorFlow / Keras
- MobileNetV2 Transfer Learning
- NumPy and Pillow
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit
- Grad-CAM Explainable AI

## Project Structure

```text
Rice-Leaf-Disease-Detection/
├── app.py
├── src/
│   ├── inspect_dataset.py
│   ├── visualize_dataset.py
│   ├── split_dataset.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── gradcam.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── best_model.keras
│   ├── final_model.keras
│   └── class_names.json
├── outputs/
├── requirements.txt
├── .gitignore
└── README.md
```

## Features

- Image preprocessing and dataset inspection
- Training, validation, and test dataset organization
- MobileNetV2-based transfer learning
- Data augmentation during training
- Early stopping and model checkpointing
- Accuracy, precision, recall, F1-score, and confusion matrix evaluation
- Grad-CAM heatmap and image overlay
- Interactive Streamlit web application

## Setup Instructions

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Rice-Leaf-Disease-Detection
```

### 2. Create a virtual environment

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Prepare the dataset

Download the rice leaf dataset from its original source and organize the images by class under:

```text
data/raw/Rice_Leaf_AUG/
```

Run the inspection and split scripts:

```bash
python src/inspect_dataset.py
python src/visualize_dataset.py
python src/split_dataset.py
```

### 5. Train the model

```bash
python src/train_model.py
```

The trained model and class mapping are saved under `models/`.

### 6. Evaluate the model

```bash
python src/evaluate_model.py
```

### 7. Run the Grad-CAM visualization

```bash
python src/gradcam.py
```

### 8. Launch the web application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

## Model Evaluation

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Actual evaluation results should be recorded after running the evaluation script on the held-out test set.

## Explainable AI

Grad-CAM highlights image regions that influence the model's class score. These visualizations help inspect model behavior but do not prove that the model has learned the correct disease symptoms.

## Limitations

- Predictions depend on the dataset and training procedure.
- Performance on real-world photographs may differ from performance on the test set.
- Class scores are not necessarily calibrated disease probabilities.
- The application is an educational prototype, not a replacement for expert agricultural diagnosis.
- Dataset duplicates and augmented images should be checked to prevent data leakage.

## Future Improvements

- Group-aware splitting of original and augmented images
- Fine-tuning selected MobileNetV2 layers
- Model calibration and additional robustness testing
- Testing on independent field photographs
- Deployment to a suitable hosting platform

