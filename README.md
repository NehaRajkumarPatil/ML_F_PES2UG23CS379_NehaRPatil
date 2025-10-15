# Automated Detection of Steel Surface Defects using Machine Learning & Computer Vision

## 📌 Project Overview
Defects on steel surfaces — such as **scratches, inclusions, pitting, rolled-in scales, and patches** — reduce the quality, durability, and value of steel products.  
Manual inspection is time-consuming, inconsistent, and costly.  
This project leverages **deep learning and computer vision** to automatically detect surface defects from steel images.

---

## 🎯 Objectives
- Automate defect classification from steel surface images.
- Improve accuracy and consistency of quality inspection.
- Provide a deployable deep learning model for real-world usage.

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

---

## 🛠️ Tech Stack
- **Python**
- **TensorFlow / Keras**
- **NumPy, Pandas**
- **Matplotlib, Seaborn**
- **Scikit-learn**

---

## ⚙️ Methodology
1. **Data Preprocessing**
   - Image resizing to `(256, 256)`.
   - Train-validation-test split.
   - Data augmentation for robust learning.

2. **Model Architecture**
   - **Convolutional Neural Network (CNN)** built with Keras Sequential API.
   - Layers extract hierarchical features:
     - Early layers detect edges & textures.
     - Deeper layers identify cracks, patches, and complex patterns.
   - Optimizer: **Adam (lr = 0.001)**.
   - Loss: **Categorical Crossentropy**.

3. **Training**
   - Up to **25 epochs**.
   - **Callbacks**:
     - Early stopping (prevents overfitting).
     - Model checkpoint (saves best model).

4. **Evaluation**
   - Accuracy and loss curves.
   - Confusion matrix to analyze per-class performance.

5. **Deployment**
   - Model saved as:  
     - `final_steel_defect_model.keras`  
   - Supports predictions on:
     - Entire test dataset.
     - Single input image.

---

## 📊 Results
- **Accuracy**: Training and validation accuracies improved steadily and stabilized.
- **Confusion Matrix**: Showed strong performance across most defect classes, with minor confusion in visually similar categories.
- Visualizations included:
  - Training vs validation accuracy/loss plots.
  - Example predictions with true vs predicted labels.

---

## 🚀 How to Run
1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
