import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from task1 import get_datasets
from task2 import RNNModel

"""
Intro to Machine Learning Assignment 4
Encompasses the solution to Task 3.
Students: Jackie Javier, Pranitha Achanta, Robert McDaniels
"""
# HYPERPARAMETERS
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64

# Accuracy calc (should be safer?)
def compute_accuracy(y_pred, y_true):
  epsilon = 1e-3  # bigger safety buffer

  relative_error = torch.abs(y_pred - y_true) / (torch.abs(y_true) + epsilon)

  accuracy = 1 - torch.mean(relative_error)

  return accuracy.item()

# Training
def train_model(X_train, y_train, X_test, y_test, stock_name):

  model = RNNModel(input_size=1, hidden_size=HIDDEN_SIZE, output_size=1)
  loss_func = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

  train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

  train_losses = []
  test_losses = []

  start_time = time.time()

  for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for X_batch, y_batch in train_loader:
      optimizer.zero_grad()

      outputs = model(X_batch)
      loss = loss_func(outputs, y_batch)

      loss.backward()
      optimizer.step()

      epoch_loss += loss.item()

    train_losses.append(epoch_loss / len(train_loader))

    # Evaluate
    model.eval()
    with torch.no_grad():
      test_pred = model(X_test)
      test_loss = loss_func(test_pred, y_test)

    test_losses.append(test_loss.item())

    print(f"{stock_name} | Epoch {epoch+1}/{EPOCHS} | Test Loss: {test_loss.item():.6f}")

  end_time = time.time()
  total_time = end_time - start_time

  # Final metrics
  model.eval()
  with torch.no_grad():
    train_pred = model(X_train)
    test_pred = model(X_test)

    train_loss = loss_func(train_pred, y_train).item()
    test_loss = loss_func(test_pred, y_test).item()
    test_accuracy = compute_accuracy(test_pred, y_test)

  return model, train_losses, test_losses, train_loss, test_loss, test_accuracy, total_time, test_pred

# Plotting
def plot_losses(train_losses, test_losses, stock_name):
  plt.figure()
  plt.plot(train_losses, label="Train Loss")
  plt.plot(test_losses, label="Test Loss")
  plt.title(f"Loss Curve - {stock_name}")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend()
  plt.savefig("loss_" + stock_name + ".png")

def plot_predictions(y_true, y_pred, stock_name):
  plt.figure()
  plt.plot(y_true.numpy(), label="True")
  plt.plot(y_pred.numpy(), label="Predicted")
  plt.title(f"Predictions vs True - {stock_name}")
  plt.legend()
  plt.savefig("pred_" + stock_name + ".png")

if __name__ == "__main__":

  datasets = get_datasets()

  results = {}

  for stock, data in datasets.items():
    print(f"\nTraining RNN for {stock}...")

    model, train_losses, test_losses, train_loss, test_loss, test_accuracy, time_cost, test_pred = train_model(
      data["X_train"],
      data["y_train"],
      data["X_test"],
      data["y_test"],
      stock
    )

    results[stock] = {
      "train_loss": train_loss,
      "test_loss": test_loss,
      "accuracy": test_accuracy,
      "time": time_cost
    }

    print(f"\n{stock} Results:")
    print(f"Train Loss: {train_loss:.6f}")
    print(f"Test Loss : {test_loss:.6f}")
    print(f"Accuracy  : {test_accuracy*100:.2f}%")
    print(f"Time      : {time_cost:.2f} seconds")

    plot_losses(train_losses, test_losses, stock)
    plot_predictions(data["y_test"], test_pred, stock)

  print("\nFinal Summary:")
  for stock, res in results.items():
    print(f"{stock}: Accuracy={res['accuracy']*100:.2f}%, Test Loss={res['test_loss']:.6f}")
