# PyTorch Regression: Medical Insurance Bill Predictor

This directory contains the PyTorch implementation of a regression task to predict medical insurance costs based on personal attributes.

## 1. Dataset Description
The model uses the **Medical Cost Personal Dataset** (`insurance.csv`). It contains 1,338 records with the following features:
* **age**: Age of primary beneficiary
* **sex**: Gender (male/female)
* **bmi**: Body mass index
* **children**: Number of children covered by health insurance
* **smoker**: Smoking status (yes/no)
* **region**: US geographic region
* **charges**: Individual medical costs billed by health insurance (Target Variable)

## 2. Preprocessing Steps
* **Categorical Encoding**: `sex`, `smoker`, and `region` are encoded using One-Hot Encoding (`pd.get_dummies` with `drop_first=True` to avoid multicollinearity).
* **Feature Scaling**: Numerical features are standardized using `StandardScaler` to ensure zero mean and unit variance.
* **Data Splitting**: The dataset is split into 80% training and 20% testing sets using `train_test_split`.
* **Tensor Conversion**: Features and targets are mapped into PyTorch `torch.float32` tensors and wrapped in a custom `Dataset` and `DataLoader`.

## 3. Model Architecture Details
A custom Feedforward Neural Network (`RegressionNN`) is implemented using PyTorch:
* **Input Layer**: Dimension matches the processed feature count.
* **Hidden Layer 1**: Linear(input_size, 64) -> ReLU()
* **Hidden Layer 2**: Linear(64, 32) -> ReLU()
* **Output Layer**: Linear(32, 1) (Single continuous output for prediction)

## 4. Evaluation Metrics and Result Discussion
The model was trained for 150 epochs using the Adam optimizer (`lr=0.01`) and Mean Squared Error (MSE) loss.
Performance on the test set:
* **Mean Absolute Error (MAE)**: `2,769.79`
* **Mean Squared Error (MSE)**: `20,713,455.15`
* **R² Score**: `0.8666`

**Discussion:** With an R² score of ~0.867, the model successfully explains roughly 86.7% of the variance in the insurance charges. The MAE indicates that the model's predictions are, on average, within ~$2,769 of the actual billed amount, which is strong given the high variance of U.S. medical costs due to factors like varying treatments or emergencies unseen in the basic feature set.

## 5. Instructions to Run the Project
1. Ensure you have Python installed.
2. Install the required dependencies:
   ```bash
   pip install torch pandas scikit-learn matplotlib numpy
   ```
3. Run the training script:
   ```bash
   python train_regression.py
   ```
4. The script will train the model, output evaluation metrics to the console, and save both the training loss plot (`training_loss.png`) and the model states (`model.pth`).
