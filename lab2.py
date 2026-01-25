import torch
from torch import nn
from sklearn.datasets import make_moons
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

device = "cuda" if torch.cuda.is_available() else "cpu"

RANDOM_SEED = 42

X_moon, y_moon = make_moons(1000, noise=0.1, random_state=RANDOM_SEED)

X_moon = torch.from_numpy(X_moon).type(torch.float)
y_moon = torch.from_numpy(y_moon).type(torch.float)

# plt.scatter(X_moon[:, 0], X_moon[:, 1], c=y_moon, cmap=plt.cm.RdYlBu)
# plt.show()


X_train, X_test, y_train, y_test = train_test_split(
    X_moon, y_moon, test_size=0.2, random_state=RANDOM_SEED
)

X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)


class Moon(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU,
            nn.Linear(32, 32),
            nn.ReLU,
            nn.Linear(32, 32),
            nn.ReLU,
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.linear_stack(x)


model = Moon().to(device)
