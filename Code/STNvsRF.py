import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

# === Load and Preprocess CSV Dataset ===
df = pd.read_csv(r"C:\Users\jrohi\OneDrive\Desktop\Proj\Dataset\user_a.csv")

X = df.iloc[:, :-1].values  # Features
y = df.iloc[:, -1].values   # Labels

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split and convert to PyTorch tensors
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

# === Define STN Model ===
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
        # Initialize as identity transformation
        self.localization[4].weight.data.zero_()
        self.localization[4].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        theta = self.localization(x)  # not used in linear features directly
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# === Training Function ===
def train(model, train_loader, criterion, optimizer, epochs=20):
    acc_list = []
    model.train()
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
        print(f"Epoch {epoch+1:2d} Accuracy: {acc*100:.2f}%")
    return acc_list

# === Train STN Model ===
print("🚀 Training STN model...")
stn_model = STN(input_dim, num_classes)
criterion = nn.CrossEntropyLoss()
stn_optimizer = optim.Adam(stn_model.parameters(), lr=0.001)
stn_acc = train(stn_model, train_loader, criterion, stn_optimizer, epochs=20)

# === Train Random Forest ===
print("\n🌲 Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)
print(f"✅ Random Forest Accuracy: {rf_acc*100:.2f}%")

# === Print and Compare Results ===
print("\n📊 STN Accuracy per Epoch:")
for i, acc in enumerate(stn_acc):
    print(f"Epoch {i+1:2d}: {acc*100:.2f}%")

print(f"\n🎯 Final STN Accuracy: {stn_acc[-1]*100:.2f}%")
print(f"🎯 Random Forest Accuracy: {rf_acc*100:.2f}%")

# === Plot Bar Graph ===
models = ['STN (Epoch 20)', 'Random Forest']
accuracies = [stn_acc[-1] * 100, rf_acc * 100]

plt.bar(models, accuracies, color=['cornflowerblue', 'seagreen'])
plt.ylabel('Accuracy (%)')
plt.title('STN vs Random Forest on Custom Dataset')
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
