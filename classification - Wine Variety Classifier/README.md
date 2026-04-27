# PyTorch Classification: Wine Variety Classifier

This directory contains the PyTorch implementation of a multiclass classification task, classifying types of wine based on chemical analysis.

## 1. Dataset Description
The model utilizes the **Wine recognition dataset** from scikit-learn. This dataset is the result of a chemical analysis of wines grown in the same region in Italy by three different cultivators.
* **Features**: Alcohol, Malic acid, Ash, Alcalinity of ash, Magnesium, Total phenols, Flavanoids, Nonflavanoid phenols, Proanthocyanins, Color intensity, Hue, OD280/OD315 of diluted wines, and Proline (**13 continuous numerical variables**).
* **Target**: Categorical variable corresponding to the wine variety (**3 classes**).

## 2. Preprocessing Steps
* **Data Retrieval**: Loaded using `sklearn.datasets.load_wine`.
* **Feature Scaling**: All 13 chemical features are normalized using `StandardScaler` to ensure they contribute equally to the model's learning process.
* **Data Splitting**: Standard 80/20 train/test split utilizing `train_test_split`.
* **Tensor Conversion**: Data is converted to PyTorch Tensors (`torch.float32` for features and `torch.long` for labels) and managed via `DataLoader` with batching.

## 3. Model Architecture Details
The script builds a Feedforward Neural Network (`ClassificationNN`) designed for high-dimensional feature spaces.
* **Input Layer**: Size 13 (mapping to the 13 chemical features).
* **Hidden Layer 1**: Linear(13, 32) + ReLU Activation.
* **Hidden Layer 2**: Linear(32, 16) + ReLU Activation.
* **Output Layer**: Linear(16, 3) representing the logits for the 3 wine varieties. (Loss calculation uses `CrossEntropyLoss`).

## 4. Evaluation Metrics and Result Discussion
The model is trained using the **Adam optimizer** with a learning rate of `0.01` for `100` epochs.
* **Accuracy Score**: The model typically achieves very high accuracy (often 95-100%) as the chemical signatures are quite distinct.
* **Classification Report**: Provides a detailed breakdown of precision, recall, and f1-score for each wine variety.
* **Discussion**: The neural network effectively captures the non-linear relationships between chemical markers to distinguish between the three cultivators.

## 5. Instructions to Run the Project
1. Ensure your environment has the necessary libraries:
   ```bash
   pip install torch scikit-learn matplotlib numpy
   ```
2. Run the training script:
   ```bash
   py train_classification.py
   ```
3. The script will output:
   - Training logs with loss values.
   - A final evaluation report.
   - `training_loss.png`: A plot of the loss curve.
   - `model.pth`: The saved weights of the trained model.