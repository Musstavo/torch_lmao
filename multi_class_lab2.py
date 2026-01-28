import torch
from torch import nn
from sklearn.datasets import make_moons
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import numpy as np
from torchmetrics import Accuracy
from torch.utils.data import DataLoader, TensorDataset
from helper_functions import plot_predictions, plot_decision_boundary

device = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
N = 100
D = 2
K = 3
X = np.zeros((N * K, D))
y = np.zeros(N * K, dtype="uint8")

for j in range(K):
    ix = range(N * j, N * (j + 1))
    r = np.linspace(0.0, 1, N)
    t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.6
    X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
    y[ix] = j

# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
# plt.show()

acc_fn = Accuracy(task="multiclass", num_classes=4).to(device)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)

X_train = torch.from_numpy(X_train).type(torch.float)
X_test = torch.from_numpy(X_test).type(torch.float)
y_train = torch.from_numpy(y_train).type(torch.long)
y_test = torch.from_numpy(y_test).type(torch.long)

X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

train_ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True)

test_ds = TensorDataset(X_test, y_test)
test_loader = DataLoader(dataset=test_ds, batch_size=32, shuffle=True)


class Spiral(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 64)
        self.layer4 = nn.Linear(64, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        return self.layer4(x)


model = Spiral().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 0.001, weight_decay=1e-4)

epochs = 1000
for epoch in tqdm(range(epochs)):
    model.train()
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(train_loader):
        y_train_logits = model(X).squeeze(dim=1)
        y_train_pred = torch.argmax(y_train_logits, dim=1)

        loss = loss_fn(y_train_logits, y)
        acc = acc_fn(y, y_train_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss = train_loss + loss.item()
        train_acc = train_acc + acc

    train_loss = train_loss / len(train_loader)
    train_acc = train_acc / len(train_loader)

    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for X_test_batch, y_test_batch in test_loader:
            X_test_batch, y_test_batch = (
                X_test_batch.to(device),
                y_test_batch.to(device),
            )

            y_test_logits = model(X_test_batch).squeeze(dim=1)
            y_test_pred = torch.argmax(y_test_logits, dim=1)

            loss2 = loss_fn(y_test_logits, y_test_batch)
            acc2 = acc_fn(y_test_batch, y_test_pred)

            test_loss = test_loss + loss2.item()
            test_acc = test_acc + acc2

        test_loss = test_loss / len(test_loader)
        test_acc = test_acc / len(test_loader)

    if epoch % 100 == 0:
        print(
            f"Epoch: {epoch} | Loss: {train_loss:.2f} Acc: {train_acc:.2f} | Test loss: {test_loss:.2f} Test acc: {test_acc:.2f}"
        )

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model, X_train, y_train)

plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model, X_test, y_test)

plt.show()
