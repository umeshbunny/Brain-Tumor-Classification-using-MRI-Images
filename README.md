# Brain Tumor MRI Image Classification Pipeline

This repository contains an end-to-end computer vision and machine learning pipeline developed to classify brain tumors from patient MRI scan matrices into **malignant (yes)** or **benign (no)** categories.

The project systematically benchmarks six distinct classification paradigms to determine the most stable deployment model.

---

# Performance Benchmarks & Metrics

All algorithms were subjected to rigorous evaluation across primary machine learning validation matrices.

| Model | Training Accuracy | Testing Accuracy | Precision | Recall | Specificity | F1 Score | AUC |
|------|------|------|------|------|------|------|------|
| Logistic Regression | 1.000 | 0.843 | 0.848 | 0.903 | 0.750 | 0.875 | 0.887 |
| Support Vector Machine (SVM) | 0.916 | 0.784 | 0.750 | 0.968 | 0.500 | 0.845 | 0.871 |
| K-Nearest Neighbors (KNN) | 0.782 | 0.765 | 0.757 | 0.903 | 0.550 | 0.824 | 0.791 |
| Random Forest | 0.911 | 0.745 | 0.725 | 0.935 | 0.450 | 0.817 | 0.848 |
| Decision Tree | 0.881 | 0.725 | 0.793 | 0.742 | 0.700 | 0.767 | 0.736 |
| AdaBoost | 1.000 | 0.725 | 0.718 | 0.903 | 0.450 | 0.800 | 0.774 |

---

#  Selection Strategy Summary

While **Logistic Regression** scored the highest raw testing accuracy (**84.3%**), the **Support Vector Machine (SVM)** was selected as the **Final Best Model**.

The regularized SVM configuration:

```python
kernel='rbf'
C=0.7
gamma='scale'
```

successfully managed the trade-off between overfitting and generalization, establishing the strongest architectural balance between bias and variance across stratified validations.

---

#  Pipeline Architecture

## 1. Image Preprocessing & Noise Isolation

Images are parsed across structural label domains and converted via OpenCV grayscale transformation to aggressively reduce sensor artifacting and background channel noise.

```python
import cv2

image = cv2.imread(path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

---

## 2. Dimensionality Optimization

MRI scans are uniformly resized to **64 × 64** matrices to control structural parameter overfitting during dense matrix flattening operations.

```python
resized = cv2.resize(gray, (64, 64))
```

---

## 3. Data Standardization

Pixel intensity values are normalized and standardized before model training.

```python
X = X / 255.0
```

Feature scaling is further performed using:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

## 4. Dataset Splitting

The dataset is partitioned into training and testing segments using stratified randomization.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

---

#  Machine Learning Models Used

## Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
```

---

## Support Vector Machine (Final Selected Model)

```python
from sklearn.svm import SVC

svm_model = SVC(
    kernel='rbf',
    C=0.7,
    gamma='scale',
    probability=True
)

svm_model.fit(X_train, y_train)
```

---

## K-Nearest Neighbors

```python
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
```

---

## Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)
```

---

## Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

dt_model.fit(X_train, y_train)
```

---

## AdaBoost

```python
from sklearn.ensemble import AdaBoostClassifier

ab_model = AdaBoostClassifier(
    n_estimators=100,
    random_state=42
)

ab_model.fit(X_train, y_train)
```

---

#  Evaluation Metrics

The following metrics were used to evaluate all classification models:

- Accuracy
- Precision
- Recall
- Specificity
- F1 Score
- ROC-AUC Score

```python
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
```

---

#  Requirements & Technology Stack

## Languages & Environments

- Python 3
- Google Colab

## Computer Vision

- OpenCV (`cv2`)

## Data Science Libraries

- NumPy
- Pandas

## Machine Learning

- Scikit-learn

## Visualization

- Matplotlib

---

#  Project Workflow

```text
MRI Dataset
     ↓
Image Preprocessing
     ↓
Grayscale Conversion
     ↓
Image Resizing
     ↓
Normalization
     ↓
Feature Scaling
     ↓
Train/Test Split
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Best Model Selection
```

---

#  Installation

Clone the repository:

```bash
git clone https://github.com/umeshbunny/Brain-Tumor-Classification-using-MRI-Images
```

Move into the project directory:

```bash
cd brain-tumor-mri-classification
```

Install required dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn opencv-python
```

---

#  Run the Project

Execute the Python file:

```bash
python main.py
```

or run directly inside Google Colab Notebook.

---

#  Visualization Support

Performance visualizations and comparative analysis graphs are generated using Matplotlib.

```python
import matplotlib.pyplot as plt
```

Graphs include:

- Accuracy Comparison
- ROC Curves
- Precision-Recall Comparisons
- Confusion Matrix Visualizations

---

#  Research Objectives

The project was designed to:

- Improve MRI tumor classification efficiency
- Compare traditional machine learning algorithms
- Analyze overfitting and generalization behavior
- Build a stable medical imaging prediction pipeline
- Evaluate statistical classification reliability

---

#  Future Improvements

- Deep Learning CNN Architectures
- Transfer Learning Models
- MRI Segmentation Integration
- Real-Time Clinical Deployment APIs
- Explainable AI (XAI) Heatmaps
- Hybrid Ensemble Architectures

---

#  Author

Developed for educational, research, and medical image classification purposes using machine learning and computer vision methodologies.
