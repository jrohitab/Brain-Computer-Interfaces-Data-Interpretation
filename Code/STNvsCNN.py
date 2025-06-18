import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load CSV dataset
df = pd.read_csv(r"C:\Users\jrohi\OneDrive\Desktop\Proj\Dataset\user_a.csv")  # File path added
X = df.iloc[:, :-1].values  # Features
y = df.iloc[:, -1].values   # Labels

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to PyTorch tensors
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define CNN model
class CNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define STN model with a Spatial Transformer Layer
class STN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(STN, self).__init__()
        self.localization = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 6)
        )
        self.localization[4].weight.data.zero_()
        self.localization[4].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        theta = self.localization(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Training function
def train(model, train_loader, criterion, optimizer, epochs=20):
    acc_list = []
    for epoch in range(epochs):
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        acc = correct / total
        acc_list.append(acc)
    return acc_list

# Initialize models, optimizers, and loss function
input_dim = X_train.shape[1]
num_classes = len(set(y))

cnn_model = CNN(input_dim, num_classes)
stn_model = STN(input_dim, num_classes)

criterion = nn.CrossEntropyLoss()
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
stn_optimizer = optim.Adam(stn_model.parameters(), lr=0.001)

# Train both models
print("Training STN...")
stn_acc = train(stn_model, train_loader, criterion, stn_optimizer)
print("Training CNN...")
cnn_acc = train(cnn_model, train_loader, criterion, cnn_optimizer)

# Display accuracy values
print("\nSTN Accuracy for 20 epochs:")
print(stn_acc)

print("\nCNN Accuracy for 20 epochs:")
print(cnn_acc)
