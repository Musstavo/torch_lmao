import torch
from torch import nn
from sklearn.datasets import make_moons
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import numpy as py

device = "cuda" if torch.cuda.is_available() else "cpu"

RANDOM_SEED = 42

X_moon, y_moon = make_moons(1000, noise=0.3, random_state=RANDOM_SEED)

X_moon = torch.from_numpy(X_moon).type(torch.float)
y_moon = torch.from_numpy(y_moon).type(torch.float)

# plt.scatter(X_moon[:, 0], X_moon[:, 1], c=y_moon, cmap=plt.cm.RdYlBu)
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X_moon, y_moon, test_size=0.2, random_state=RANDOM_SEED
)
train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)


class Moon(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(2, 62),
            nn.ReLU(),
            nn.Linear(62, 62),
            nn.ReLU(),
            nn.Linear(62, 62),
            nn.ReLU(),
            nn.Linear(62, 1),
        )

    def forward(self, x):
        return self.linear_stack(x)


def accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


model = Moon().to(device)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), 0.1, momentum=0.9)

torch.manual_seed(42)
epochs = 1000

for epoch in tqdm(range(epochs)):
    model.train()
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        y_train_logits = model(X).squeeze()
        y_train_pred = torch.round(torch.sigmoid(y_train_logits))

        loss = loss_fn(y_train_logits, y)

        train_loss += loss.item()
        train_acc += accuracy(y, y_train_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
            test_logits = model(X_test_batch).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))

            test_loss += loss_fn(test_logits, y_test_batch)
            test_acc += accuracy(y_test_batch, test_pred)

    test_loss = test_loss / len(test_loader)
    test_acc = test_acc / len(test_loader)

    if epoch % 100 == 0:
        print(
            f"Epoch: {epoch} | Loss: {train_loss:.2f} Acc: {train_acc:.2f} | Test loss: {test_loss:.2f} Test acc: {test_acc:.2f}"
        )
