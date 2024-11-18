"""
Binary Classification using PyTorch

I created a single feature/column dataset with two seperate classes with 6 points each.
9 points are used for training and 3 are used for testing.
"""

import torch
from torch import nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Data
X = torch.tensor([[1],[2],[3],[6],[5],[7],[20],[24],[25],[27],[26],[30]], dtype=torch.float32)
y = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = nn.Sequential(
    nn.Linear(1,2),
    nn.Sigmoid(),
    nn.Linear(2,1),
    nn.Sigmoid(),
)

# Training
n_epochs = 500
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=5e-2)
loss_list = torch.empty(n_epochs)

for epoch in range(n_epochs):
  for i in range(len(X_train)):
    y_hat = model(X_train[i])
    loss = loss_fn(y_hat[0], y_train[i])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
  loss_list[epoch] = loss.item()

# Prediction/Testing
y_pred = model(X_test)

# Analysis
accuracy = sum(y_pred.detach().numpy().round().flatten() == y_test)/len(y_test)*100 # accuracy score in percentage
print(accuracy.item(),"%")
plt.plot(loss_list)
plt.title("Binary Classification Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
