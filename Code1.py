# Code for Model A -- Full dataset including all variables as predictors
# Nastassja Bibat

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder

# -------------------------------
# Data Loading and Preprocessing
# -------------------------------
df = pd.read_csv("Full dataset.csv")
df.dropna(inplace=True)

# Encode categorical columns (if any) except for the target
target = "loan_status"
for col in df.select_dtypes(include=['object']).columns:
    if col != target:
        df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop(target, axis=1).values
y = df[target].values  # Assuming binary labels: 0 (rejected) and 1 (approved)

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors, ensuring y tensors are column vectors for BCE loss.
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test  = torch.tensor(X_test, dtype=torch.float32)
y_test  = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# -------------------------------
# Define the Neural Network Model
# -------------------------------
torch.manual_seed(42)

class LoanApprovalNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LoanApprovalNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


input_dim = X_train.shape[1]
hidden_dim = 11
model = LoanApprovalNN(input_dim, hidden_dim)

# -------------------------------
# Loss Function, Optimizer, and Loss History Setup
# -------------------------------
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 8000
train_loss_history = []
test_loss_history = []

# -------------------------------
# Training the Network and Recording Losses
# -------------------------------
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass for training data
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    # Record training loss
    train_loss_history.append(loss.item())
    
    # Evaluate test loss
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        test_loss_history.append(test_loss.item())

        test_outputs = model(X_test)
        # Convert probabilities to binary predictions (threshold = 0.5)
        predicted = (test_outputs >= 0.5).float()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

# Convert tensors to numpy arrays for metric computation
y_true = y_test.numpy()
y_pred = predicted.numpy()

# Compute assessment metrics
accuracy = accuracy_score(y_true, y_pred) * 100
precision = precision_score(y_true, y_pred) * 100
recall = recall_score(y_true, y_pred) * 100
f1 = f1_score(y_true, y_pred) * 100

# In the confusion matrix for binary classification:
cm = confusion_matrix(y_true, y_pred)
TN = cm[0, 0]
FP = cm[0, 1]
specificity = (TN / (TN + FP)) * 100 if (TN + FP) > 0 else 0

print("\n--- Evaluation Metrics ---")
print(f"Accuracy  : {accuracy:.2f}")
print(f"Recall    : {recall:.2f}")
print(f"Precision : {precision:.2f}")
print(f"F1 Score  : {f1:.2f}")
print(f"Specificity: {specificity:.4f}")

# -------------------------------
# Plotting the Loss
# -------------------------------
plt.figure(figsize=(8, 6))
plt.plot(train_loss_history, label="Train Loss")
plt.plot(test_loss_history, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Test Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.show()