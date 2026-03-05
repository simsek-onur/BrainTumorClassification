# Brain Tumor Classification from MRI Images

A deep learning project that classifies brain MRI scans into **tumor** and **no-tumor** categories using convolutional neural networks built with TensorFlow/Keras.

## Overview

This project loads brain MRI image data from a MATLAB `.mat` file, preprocesses the images, and trains a CNN model to detect the presence of brain tumors. The workflow covers the full ML pipeline — from data loading and exploration through model training, evaluation, and visualization of results.

## Dataset

- **Source:** `Brain.mat` — a MATLAB data file containing brain MRI images and their corresponding labels.
- **Classes:** Binary classification (tumor vs. no tumor)
- **Format:** Images stored as arrays within the `.mat` file, loaded using `scipy.io.loadmat`

## Project Structure

```
├── BrainTumorClassification.ipynb   # Main Jupyter notebook with full pipeline
├── Brain.mat           # MRI image dataset (MATLAB format)
└── README.md
```

## Pipeline

1. **Data Loading** — Read the `.mat` file using `scipy.io` and extract image arrays and labels.
2. **Exploratory Data Analysis** — Visualize sample MRI images and inspect class distribution.
3. **Preprocessing** — Resize/normalize images and prepare train/test splits.
4. **Model Building** — Construct a CNN architecture with convolutional, pooling, and dense layers.
5. **Training** — Train the model with appropriate loss function and optimizer.
6. **Evaluation** — Assess performance using accuracy, loss curves, confusion matrix, and classification report.

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- SciPy
- TensorFlow / Keras
- scikit-learn

Install dependencies:

```bash
pip install numpy matplotlib scipy tensorflow scikit-learn
```

## Usage

1. Place `Brain.mat` in the same directory as the notebook.
2. Open and run the notebook:

```bash
jupyter notebook BrainTumorClassification.ipynb
```

3. Execute cells sequentially to reproduce the full training and evaluation pipeline.
