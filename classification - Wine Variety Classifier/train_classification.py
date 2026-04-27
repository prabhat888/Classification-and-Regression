import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. Dataset Definition
# ==========================================
class ClassificationDataset(Dataset):
    """Custom PyTorch Dataset for loading tabular classification data."""
    def __init__(self, X, y):
        # Features: float32 tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        # Target: int64 (long) tensors for CrossEntropyLoss
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==========================================
# 2. Model Architecture
# ==========================================
class ClassificationNN(nn.Module):
    """Feedforward Neural Network for Multiclass Classification."""
    def __init__(self, input_size, num_classes):
        super(ClassificationNN, self).__init__()
        # Input layer -> Linear -> ReLU -> Linear -> ReLU -> Linear -> Output
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes) # Raw logits for CrossEntropyLoss
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. Preprocessing Functions
# ==========================================
def load_and_preprocess_data():
    """Loads Wine dataset, prints stats, normalizes, and splits the data."""
    print("--- Loading Wine Dataset ---")
    data = load_wine()
    X = data.data
    y = data.target
    target_names = data.target_names

    print(f"Dataset shape: {X.shape}")
    print(f"Target classes: {target_names}")

    # Split dataset into train and test window (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize numerical features using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, target_names

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
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss (CrossEntropy): {avg_loss:.4f}")

    return epoch_losses

# ==========================================
# 5. Evaluation Function
# ==========================================
def evaluate_model(model, X_test, y_test, target_names):
    """Evaluates model performance on the test set."""
    print("\n--- Evaluation Results ---")
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        # Getting raw logits
        logits = model(X_tensor)
        # Applying argmax to get predicted classes
        _, y_pred = torch.max(logits, 1)
        y_pred = y_pred.numpy()

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy*100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

# ==========================================
# 6. Visualization Function
# ==========================================
def plot_losses(losses, save_path="training_loss.png"):
    """Plots and saves the training loss chart."""
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(losses) + 1), losses, label='Training Loss', color='green')
    plt.title('Training Loss vs. Epochs (Classification)')
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nTraining loss plot saved to '{save_path}'")

# ==========================================
# 7. Main Execution Flow
# ==========================================
def main():
    # Phase 1: Data Preprocessing
    X_train, X_test, y_train, y_test, target_names = load_and_preprocess_data()

    # Convert to PyTorch Dataset & DataLoader
    train_dataset = ClassificationDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Phase 2: Model Configuration
    input_size = X_train.shape[1]
    num_classes = len(target_names)
    model = ClassificationNN(input_size, num_classes)
    
    # Loss & Optimizer
    # CrossEntropyLoss internally applies log softmax
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Phase 3: Train Model
    num_epochs = 100 
    losses = train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs)

    # Phase 4: Evaluate
    evaluate_model(model, X_test, y_test, target_names)

    # Phase 5: Visualize
    plot_losses(losses)

    # Phase 6: Save Model
    model_path = "model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Trained classification model saved as '{model_path}'")

if __name__ == "__main__":
    main()