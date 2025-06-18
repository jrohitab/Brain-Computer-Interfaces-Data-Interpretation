import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# === Load and Preprocess CSV Dataset ===
df = pd.read_csv(r"C:\Users\jrohi\OneDrive\Desktop\Proj\Dataset\user_a.csv")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

input_dim = X_train.shape[1]
num_classes = len(np.unique(y))

# === STN Model ===
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
        self.localization[4].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        theta = self.localization(x)  # ignored in tabular, simulated
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# === Simulated Xception-style MLP for Tabular Data ===
class XceptionTabular(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(XceptionTabular, self).__init__()
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x)

# === Training Function ===
def train(model, loader, criterion, optimizer, epochs=20):
    acc_list = []
    model.train()
    for epoch in range(epochs):
        correct = 0
        total = 0
        for inputs, labels in loader:
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
        print(f"Epoch {epoch+1:2d} Accuracy: {acc*100:.2f}%")
    return acc_list

# === Initialize Models ===
stn_model = STN(input_dim, num_classes)
xception_model = XceptionTabular(input_dim, num_classes)

criterion = nn.CrossEntropyLoss()
stn_optimizer = optim.Adam(stn_model.parameters(), lr=0.001)
xception_optimizer = optim.Adam(xception_model.parameters(), lr=0.001)

# === Train STN ===
print("\n🚀 Training STN model...")
stn_acc = train(stn_model, train_loader, criterion, stn_optimizer, epochs=20)

# === Train Xception ===
print("\n⚡ Training Xception-style model...")
xcp_acc = train(xception_model, train_loader, criterion, xception_optimizer, epochs=20)

# === Evaluate on Test Set ===
def evaluate(model):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        acc = accuracy_score(y_test_tensor.numpy(), predicted.numpy())
        return acc

stn_test_acc = evaluate(stn_model)
xcp_test_acc = evaluate(xception_model)

# === Report Results ===
print("\n📊 STN Accuracy per Epoch:")
for i, acc in enumerate(stn_acc):
    print(f"Epoch {i+1:2d}: {acc*100:.2f}%")

print("\n📊 Xception Accuracy per Epoch:")
for i, acc in enumerate(xcp_acc):
    print(f"Epoch {i+1:2d}: {acc*100:.2f}%")

print(f"\n✅ Final STN Test Accuracy:      {stn_test_acc*100:.2f}%")
print(f"✅ Final Xception Test Accuracy: {xcp_test_acc*100:.2f}%")

# === Plot Accuracy Comparison ===
models = ['STN', 'Xception']
final_accuracies = [stn_test_acc * 100, xcp_test_acc * 100]

plt.bar(models, final_accuracies, color=['skyblue', 'orange'])
plt.ylabel('Test Accuracy (%)')
plt.title('STN vs Xception on Tabular Data')
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
