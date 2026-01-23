import torch
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

weight = 0.7
bias = 0.3

X = torch.arange(0, 1, 0.02).unsqueeze(dim=1)
y = weight * X + bias

eighty_prec_X = int(0.8 * len(X))
X_train, y_train = X[:eighty_prec_X], X[:eighty_prec_X]
X_test, y_test = X[eighty_prec_X:], X[eighty_prec_X:]


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, X):
        return self.linear_layer(X)


torch.manual_seed(42)
model = LinearRegression()
model.to(device)

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

epochs = 1000

X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

for epoch in range(epochs):
    model.train()
    train_pred = model(X_train)
    loss = loss_fn(train_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_pred = model(X_test)
        test_loss = loss_fn(test_pred, y_test)

model.eval()
with torch.inference_mode():
    y_pred = model(X_test)
print(y_pred)
