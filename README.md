# Automated Detection of Steel Surface Defects
---

## 🧱 Problem Statement

Steel manufacturing and processing often produce surfaces that can have defects such as scratches, inclusions, pitting, rolled-in scales, or patches. These defects:

- degrade the mechanical strength and durability of steel parts,  
- increase waste and rework,  
- pose risks in structural applications if left undetected,  
- require inspection that is typically manual, slow, and prone to human error.

Therefore, the core problem is:

> **How can we develop an automated, reliable, and efficient system to detect, classify, and localize surface defects on steel components using machine learning and computer vision techniques?**

The solution should minimize false positives/negatives, handle variability in defect types and background textures, and be deployable for industrial use.  

Existing research shows that defect detection in steel is challenging due to:

- high variation in defect size, shape, and contrast,  
- small defect areas that are hard to distinguish from background,  
- need for high throughput (speed) in production settings,  
- difficulty in obtaining large, balanced datasets with labeling.

Deep learning, especially CNNs, offers promise in automatically learning discriminative features and scaling detection tasks. This project aims to leverage these advances to solve the problem above in the context of steel surface defects.

---

## 📖 Project Overview

This project implements a convolutional neural network (CNN) based solution to classify and detect various kinds of surface defects on steel plates. The key goals are:

- Automate defect classification into categories like scratch, inclusion, pitting, rolled-in scale, patch.  
- Evaluate the performance and robustness of the trained model.  
- Provide an inference pipeline for new images (single or batches).  
- Lay groundwork for deployment in industrial settings.

---


- **data/** — dataset organized by class for train / validation / test  
- **notebooks/** — Jupyter notebook(s) for exploration and experimentation  
- **models/** — saved model files  
- **scripts/** — modularized scripts for training and prediction  
- **requirements.txt** — list of dependencies  
- **README.md** — project documentation  

---

## 📂 Dataset
- Dataset structured in class-wise folders:
  - **Scratches**
  - **Crazing**
  - **Inclusion**
  - **Pitting**
  - **Rolled**
  - **Patches**
- Images are resized to **256x256** before training.
- Data augmentation applied (rotation, flipping, scaling, etc.) to enhance generalization.
- Data augmentation (rotations, flips, zooms, etc.) is applied to increase variability and robustness.  
- Split into training, validation, and test sets.

---

## 🧠 Methodology

### 1. Data Preprocessing

- Load images and labels based on folder structure.  
- Split into train / validation / test subsets.  
- Resize images to a standard fixed size (e.g. 256×256).  
- Apply augmentation (flip, rotation, zoom, shift, contrast) on training data.

### 2. Model Architecture

- A **CNN model** built using Keras Sequential API.  
- Typical architecture includes:
  - Convolutional + pooling layers to extract spatial features  
  - Possibly dropout or batch normalization  
  - Dense layers culminating in a **softmax** output for multi-class classification  

- Loss Function: **Categorical Crossentropy**  
- Optimizer: **Adam** (or similar)  
- Learning rate and hyperparameters tuned via experiments.

### 3. Training

- Train over a number of epochs (e.g. up to 25), with **EarlyStopping** to avoid overfitting.  
- Use **ModelCheckpoint** to save the best model (lowest validation loss).  
- Monitor training and validation accuracy/loss curves.

### 4. Evaluation & Validation

- Plot training vs validation accuracy and loss.  
- Generate a **confusion matrix** to inspect performance class-wise.  
- Optionally compute precision, recall, F1-score for each class.  
- Display sample predictions (true vs predicted) for qualitative validation.

### 5. Inference / Prediction

- Load the saved model (`final_steel_defect_model.keras`).  
- Use it to predict:
  - Entire test set (batch inference).  
  - Single input image (preprocess → feed into model → map class).

---

## 📊 Results & Observations

- Training and validation curves converge, with limited overfitting if augmentation + early stopping used.  
- Accuracy is acceptable across most classes; however, classes with subtle visual differences may show confusion.  
- The confusion matrix helps identify problematic classes (e.g. patch vs inclusion).  
- Visual examination of predictions shows whether the model is robust or misclassifying borderline cases.  
- Based on results, one might consider more data, better architectures, or transfer learning for improvements.

---

**Project by**: Neha R. Patil  and Nandani 

## 🚀 How to Run

1. **Clone the repository**

   ```bash
   git clone https://github.com/NehaRajkumarPatil/ML_F_PES2UG23CS379_NehaRPatil.git
   cd ML_F_PES2UG23CS379_NehaRPatil
2. **Run the python code**

    ```bash
   python steel_defect.py

