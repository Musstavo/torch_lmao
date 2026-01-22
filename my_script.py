# import torch
# from torch import nn
# import matplotlib.pyplot as plt
# from pathlib import Path
#
#
# weight = 0.7
# bias = 0.3
#
# start = 0
# end = 1
# step = 0.02
# X = torch.arange(start, end, step).unsqueeze(dim=1)
# y = weight * X + bias
#
# train_split = int(0.8 * len(X))
# X_train, y_train = X[:train_split], y[:train_split]
#
# X_test, y_test = X[train_split:], y[train_split:]
#
#
# class LinearRegressionModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.weights = nn.Parameter(
#             torch.randn(1, dtype=torch.float), requires_grad=True
#         )
#         self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
#
#     def forward(self, x):
#         return self.weights * x + self.bias
#
#
# def plot_predictions(
#     train_data=X_train,
#     train_labels=y_train,
#     test_data=X_test,
#     test_labels=y_test,
#     predictions=None,
# ):
#     """
#     Plots training data, test data and compares predictions.
#     """
#     plt.figure(figsize=(10, 7))
#
#     # Plot training data in blue
#     plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
#
#     # Plot test data in green
#     plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")
#
#     if predictions is not None:
#         # Plot the predictions in red (predictions were made on the test data)
#         plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
#
#     # Show the legend
#     plt.legend(prop={"size": 14})
#
#     plt.show()
#
#
# torch.manual_seed(42)
#
# model_0 = LinearRegressionModel()
#
# loss_fn = nn.L1Loss()
#
# optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)
# train_loss_values = []
# test_loss_values = []
# epoch_count = []
# for epoch in range(100):
#     model_0.train()
#     y_pred = model_0(X_train)
#     loss = loss_fn(y_pred, y_train)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     model_0.eval
#     with torch.inference_mode():
#         test_pred = model_0(X_test)
#         test_loss = loss_fn(test_pred, y_test.type(torch.float))
#
#         if epoch % 10 == 0:
#             epoch_count.append(epoch)
#             test_loss_values.append(loss.detach().numpy())
#             train_loss_values.append(test_loss.detach().numpy())
#
# model_0.eval()
#
# with torch.inference_mode():
#     y_preds = model_0(X_test)
#
# MODEL_PATH = Path("models")
# MODEL_PATH.mkdir(parents=True, exist_ok=True)
#
# MODEL_NAME = "01_pytorch_workflow_model_0.pth"
# MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
#
# loaded_model_0 = LinearRegressionModel()
# loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
# loaded_model_0.eval()
#
# with torch.inference_mode():
#     loaded_model_preds = loaded_model_0(X_test)
# print(y_preds == loaded_model_preds)

import torch
from torch import nn
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

x = torch.arange(start, end, step).unsqueeze(dim=1)
