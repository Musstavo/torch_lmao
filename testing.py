from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from pathlib import Path
import requests

torch.manual_seed(42)

if Path("helper_functions.py").is_file():
    pass
else:
    print("Downloading helper_functions.py")
    request = requests.get(
        "https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py"
    )
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary

n_samples = 1000
device = "cuda" if torch.cuda.is_available() else "cpu"

X, y = make_circles(n_samples, noise=0.03, random_state=42)

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=10)
        self.layer_4 = nn.Linear(in_features=10, out_features=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.layer_2(x)
        x = self.relu(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        return x


model = LinearModel().to(device)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), 0.1)


def accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


epochs = 1000
for epoch in range(epochs):
    model.train()
    y_train_logits = model(X_train).squeeze()
    y_train_pred = torch.round(torch.sigmoid(y_train_logits))
    loss = loss_fn(y_train_logits, y_train)
    acc = accuracy(y_train, y_train_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        y_test_logits = model(X_test).squeeze()
        y_test_pred = torch.round(torch.sigmoid(y_test_logits))
        test_loss = loss_fn(y_test_logits, y_test)
        test_acc = accuracy(y_test, y_test_pred)
#
#     if epoch % 100 == 0:
#         print(
#             f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%"
#         )

new_x = torch.tensor([[0.3, 0.3]]).to(device)

model.eval()
with torch.inference_mode():
    new_logit = model(new_x)

    new_prob = torch.sigmoid(new_logit)
    new_pred = torch.round(new_prob)

print(f"Coordinate: {new_x.tolist()}")
print(f"Prediction: {new_pred.item()} (1 = Blue/Inner, 0 = Red/Outer)")
plt.figure(figsize=(6, 6))
plot_decision_boundary(model, X_test, y_test)
plt.scatter(0.3, 0.3, c="green", s=200, marker="x", label="New Point")
plt.legend()
plt.show()
