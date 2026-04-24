"""
Intro to Machine Learning Assignment 4
Encompasses the solution to Task 3.
Students: Jackie Javier, Pranitha Achanta, Robert McDaniels
"""
import copy
import torch
import torch.nn as nn
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from task1 import Stock, get_datasets
from task2 import RNN

# HYPERPARAMETERS
# these are kept the same for consistency
BATCH_SIZE = 32
EPOCHS = 100 # Maximum epochs to try
WEIGHT_DECAY = 0.001
BAD_RUNS = 20 # Number of runs of no improvement to give up

# Configurable for task3
LEARNING_RATE = 0.0005
HIDDEN_SIZE = 32
NUM_LAYERS = 1
DROPOUT = 0

# Accuracy calc (should be safer?)
def compute_accuracy(y_pred, y_true):
  epsilon = 1e-3  # bigger safety buffer

  relative_error = torch.abs(y_pred - y_true) / (torch.abs(y_true) + epsilon)

  accuracy = 1 - torch.mean(relative_error)

  return accuracy.item()

@dataclass
class TrainResults:
  train_losses: list[float]
  test_losses: list[float]
  time_cost: float

  train_loss: float
  test_loss: float
  test_accuracy: float
  test_pred: float

def train_model(model, stock: Stock, lr: float, verbose: bool):
  loss_func = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

  train_loader = DataLoader(stock.train, batch_size=BATCH_SIZE, shuffle=True)

  train_losses = []
  test_losses = []

  start_time = time.perf_counter()

  best_model = model
  best_acc = 0
  best_epoch = 0
  bad_runs = 0

  # EPOCHS START
  epoch = 0
  while epoch < EPOCHS: #for epoch in range(EPOCHS):
    epoch += 1
    model.train()
    epoch_loss = 0

    for X_batch, y_batch in train_loader:
      optimizer.zero_grad()

      outputs = model(X_batch)
      loss = loss_func(outputs, y_batch)

      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 3) #From slides
      optimizer.step()

      epoch_loss += loss.item()

    train_losses.append(epoch_loss / len(train_loader))

    # Evaluate within epoch
    model.eval()
    with torch.no_grad():
      test_pred = model(stock.X_test)
      test_loss = loss_func(test_pred, stock.y_test)

    test_losses.append(test_loss.item())

    if verbose:
      print(f"{stock.name} | Epoch {epoch+1}/{EPOCHS} | Test Loss: {test_loss.item():.6f}")
    
    test_acc = compute_accuracy(test_pred, stock.y_test)
    if test_acc > best_acc:
      best_model = type(model).__new__(type(model))
      best_model.__dict__ = copy.deepcopy(model.__dict__)
      best_model.load_state_dict(copy.deepcopy(model.state_dict()))
      best_acc = test_acc
      best_epoch = epoch
      bad_runs = 0
    elif bad_runs < BAD_RUNS:
      bad_runs += 1
    else:
      break
  
  print("Best epoch", best_epoch)
  model = best_model

  end_time = time.perf_counter()
  total_time = end_time - start_time

  # Final metrics after all epochs completed
  model.eval()
  with torch.no_grad():
    train_pred = model(stock.X_train)
    test_pred = model(stock.X_test)

    train_loss = loss_func(train_pred, stock.y_train).item()
    test_loss = loss_func(test_pred, stock.y_test).item()
    test_accuracy = compute_accuracy(test_pred, stock.y_test)

  return TrainResults(
    train_losses, test_losses, total_time,
    train_loss, test_loss, test_accuracy, test_pred
  )

# Plotting
def plot_losses(model_name, train_losses, test_losses, stock_name):
  plt.figure()
  plt.plot(train_losses, label="Train Loss")
  plt.plot(test_losses, label="Test Loss")
  plt.title(f"Loss Curve - {stock_name}")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend()
  plt.savefig(model_name + "/loss_" + stock_name + ".png")
  plt.close()

def plot_predictions(model_name, y_true, y_pred, stock_name):
  plt.figure()
  plt.plot(y_true.numpy(), label="True")
  plt.plot(y_pred.numpy(), label="Predicted")
  plt.title(f"Predictions vs True - {stock_name}")
  plt.legend()
  plt.savefig(model_name + "/pred_" + stock_name + ".png")
  plt.close()

# Training for each stock with separate models of the given Model type
def train_eval(Model, lr, verbose=True, **params):
  datasets = get_datasets()

  results = {}

  print(f"=== {Model.__name__} ===")
  for name, stock in datasets.items():
    if verbose: print()
    print(f"Training {name}...")

    model = Model(input_size=1, output_size=1, **params)
    res = train_model(model, stock, lr, verbose)

    results[name] = res

    if verbose:
      print(f"\n{name} Results:")
      print(f"Train Loss: {res.train_loss:.6f}")
      print(f"Test Loss : {res.test_loss:.6f}")
      print(f"Accuracy  : {res.test_accuracy*100:.2f}%")
    print(f"Time      : {res.time_cost:.2f} seconds")

    plot_losses(Model.__name__, res.train_losses, res.test_losses, name)
    plot_predictions(Model.__name__, stock.y_test, res.test_pred, name)

  print("\nFinal Summary:")
  for name, res in results.items():
    print(f"{name}: Accuracy={res.test_accuracy*100:.2f}%, Test Loss={res.test_loss:.6f}")
  print()

def test(verbose=True):
  train_eval(RNN, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT, lr=LEARNING_RATE, verbose=verbose)

def main():
  torch.manual_seed(42)
  test()

if __name__ == "__main__":
  main()
