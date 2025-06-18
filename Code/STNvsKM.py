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
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.stats import mode

# === Load and Preprocess Dataset ===
df = pd.read_csv(r"C:\Users\jrohi\OneDrive\Desktop\Proj\Dataset\user_a.csv")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

input_dim = X_train.shape[1]
num_classes = len(np.unique(y))

# === Define STN Model ===
class STN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(STN, self).__init__()
        self.localization = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )
        self.localization[4].weight.data.zero_()
        self.localization[4].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        theta = self.localization(x)  # Not used directly for tabular data
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# === Train STN ===
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

# === Evaluate STN on Test Data ===
def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        _, preds = torch.max(outputs, 1)
        return accuracy_score(y.numpy(), preds.numpy())

# === STN Training ===
print("\n🚀 Training STN model...")
stn_model = STN(input_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(stn_model.parameters(), lr=0.001)
stn_acc = train(stn_model, train_loader, criterion, optimizer, epochs=20)

# === Final STN Accuracy on Test Set ===
stn_test_acc = evaluate(stn_model, X_test_tensor, y_test_tensor)
print(f"\n✅ Final STN Test Accuracy: {stn_test_acc*100:.2f}%")

# === KMeans Clustering ===
print("\n🔍 Running KMeans clustering...")
kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_test)

# === Map KMeans Clusters to Labels Using Majority Voting ===
def map_clusters_to_labels(clusters, true_labels):
    label_map = {}
    for i in range(num_classes):
        mask = clusters == i
        if np.any(mask):
            most_common = mode(true_labels[mask])[0][0]
            label_map[i] = most_common
    return np.array([label_map[c] for c in clusters])

mapped_preds = map_clusters_to_labels(clusters, y_test)
kmeans_acc = accuracy_score(y_test, mapped_preds)
print(f"✅ KMeans Accuracy (mapped): {kmeans_acc*100:.2f}%")

# === Plot Final Comparison ===
models = ['STN', 'KMeans']
accuracies = [stn_test_acc * 100, kmeans_acc * 100]

plt.bar(models, accuracies, color=['skyblue', 'salmon'])
plt.title("STN vs KMeans on Tabular Data")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()
