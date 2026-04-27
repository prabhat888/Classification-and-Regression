import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ==========================================
# 1. Dataset Definition
# ==========================================
class InsuranceDataset(Dataset):
    """Custom PyTorch Dataset for loading tabular data."""
    def __init__(self, X, y):
        # Convert arrays to float32 tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        # Reshape y to be a column vector
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==========================================
# 2. Model Architecture
# ==========================================
class RegressionNN(nn.Module):
    """Feedforward Neural Network for Regression."""
    def __init__(self, input_size):
        super(RegressionNN, self).__init__()
        # Input layer -> Linear -> ReLU -> Linear -> ReLU -> Linear -> Output
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Single output for regression
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. Preprocessing Functions
# ==========================================
def load_and_preprocess_data(filepath):
    """Loads CSV, prints stats, encodes, normalizes, and splits the data."""
    print("--- Loading Dataset ---")
    df = pd.read_csv(filepath)
    
    # Show basic info and summary
    print(df.info())
    print("\n--- Summary Statistics ---")
    print(df.describe())

    # Handle categorical variables (sex, smoker, region) using one-hot encoding
    # drop_first=True avoids the dummy variable trap
    df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

    # Separate features and target
    X = df.drop('charges', axis=1).values
    y = df['charges'].values

    # Split dataset into train and test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# ==========================================
# 4. Training Function
# ==========================================
def train_model(model, dataloader, criterion, optimizer, num_epochs=100):
    """Trains the model and returns a list of losses across epochs."""
    print("\n--- Starting Training ---")
    model.train()
    epoch_losses = []

    for epoch in range(num_epochs):
        batch_losses = []
        for batch_X, batch_y in dataloader:
            # Forward pass
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)

            # Backward pass & optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        # Average loss for the epoch
        avg_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss (MSE): {avg_loss:,.4f}")

    return epoch_losses

# ==========================================
# 5. Evaluation Function
# ==========================================
def evaluate_model(model, X_test, y_test):
    """Evaluates model performance on the test set."""
    print("\n--- Evaluation Results ---")
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_pred = model(X_tensor).numpy()

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error (MAE): {mae:,.2f}")
    print(f"Mean Squared Error (MSE):  {mse:,.2f}")
    print(f"R² Score:                  {r2:.4f}")

# ==========================================
# 6. Visualization Function
# ==========================================
def plot_losses(losses, save_path="training_loss.png"):
    """Plots and saves the training loss chart."""
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(losses) + 1), losses, label='Training Loss', color='blue')
    plt.title('Training Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nTraining loss plot saved to '{save_path}'")

# ==========================================
# 7. Main Execution Flow
# ==========================================
def main():
    filepath = 'insurance.csv'
    
    # Check if dataset exists
    if not os.path.exists(filepath):
        print(f"Error: Could not find '{filepath}'. Please ensure it is in the same directory.")
        return

    # Phase 1: Data Preprocessing
    X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath)

    # Convert to PyTorch Dataset & DataLoader
    train_dataset = InsuranceDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Phase 2: Model Configuration
    input_size = X_train.shape[1]
    model = RegressionNN(input_size)
    
    # Loss & Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Phase 3: Train Model
    num_epochs = 150 # Training for more than the mandatory 50 epochs to converge cleanly
    losses = train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs)

    # Phase 4: Evaluate
    evaluate_model(model, X_test, y_test)

    # Phase 5: Visualize
    plot_losses(losses)

    # Phase 6: Save Model
    model_path = "model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Trained model saved as '{model_path}'")

if __name__ == "__main__":
    main()
