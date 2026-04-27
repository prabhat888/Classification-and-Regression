# PyTorch Implementation for Regression and Classification Tasks

This repository contains the assignment submission for building PyTorch implementations of both a regression and a classification task. 

## 📂 Repository Structure

The assignment is divided into two distinct projects:

### 1. [Regression Task: Medical Insurance Bill Predictor](./regression%20-%20Medical%20insurance%20bill%20predictor)
* **Dataset:** Medical Cost Personal Dataset (`insurance.csv`).
* **Objective:** Predict the individual medical costs billed by health insurance based on personal attributes (age, bmi, children, smoker, region, etc.).
* **Model:** Feedforward Neural Network (Linear -> ReLU -> Linear -> ReLU -> Linear).
* **Metrics:** Mean Absolute Error (MAE), Mean Squared Error (MSE), R² Score.

### 2. [Classification Task: Wine Variety Classifier](./classification%20-%20Wine%20Variety%20Classifier)
* **Dataset:** Wine Recognition Dataset (loaded via `sklearn.datasets`).
* **Objective:** Classify types of wine into three varieties based on chemical analysis.
* **Model:** Multiclass Feedforward Neural Network (Linear -> ReLU -> Linear -> ReLU -> Linear -> CrossEntropyLoss).
* **Metrics:** Accuracy Score, Precision, Recall, and F1-Score.

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/ashu529/Classification-and-Regression-.git
   cd Classification-and-Regression-
   ```

2. Install the necessary dependencies (applies to both tasks):
   ```bash
   pip install torch pandas scikit-learn matplotlib numpy
   ```

3. **To run the Regression Task:**
   ```bash
   cd "regression - Medical insurance bill predictor"
   py train_regression.py
   ```

4. **To run the Classification Task:**
   ```bash
   cd "classification - Wine Variety Classifier"
   py train_classification.py
   ```

## 📝 Assignment Requirements Met
- [x] PyTorch implementation for one regression task.
- [x] PyTorch implementation for one classification task.
- [x] Dataset description and preprocessing steps (detailed in individual READMEs).
- [x] Model architecture details.
- [x] Training and testing code.
- [x] Evaluation metrics and result discussion.
- [x] README file with proper instructions to run the project.
