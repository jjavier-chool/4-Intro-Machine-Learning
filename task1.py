"""
Intro to Machine Learning Assignment 4
Encompasses the solution to Task 1.
Students: Jackie Javier, Pranitha Achanta, Robert McDaniels
"""
from functools import lru_cache
from torch.utils.data import TensorDataset
import yfinance as yf
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

STOCKS = ["AAPL", "MSFT", "GOOG", "AMZN"]
# Taken from the suggested dates
START_DATE = "2010-01-01"
END_DATE = "2021-12-31"

M = 60  # number of past days (input sequence length)
N = 1   # number of future days to predict

TEST_SIZE = 0.2
RANDOM_STATE = 42

# Download the data
def download_stock_data(stock: str):
  data = yf.download(stock, start=START_DATE, end=END_DATE)
  return data['Close'].values.reshape(-1, 1)

# Convert time series into supervised learning format
def create_sequences(data, M: int, N: int):
  """
  X: past M days
  y: next N days
  """
  X, y = [], []

  for i in range(len(data) - M - N):
    X.append(data[i:i+M])
    y.append(data[i+M])

  return np.array(X), np.array(y)

# Preprocess + split
@lru_cache
def prepare_data(stock: str):
  # Download
  raw_data = download_stock_data(stock)

  # Normalize
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaled_data = scaler.fit_transform(raw_data)

  # Create sequences
  X, y = create_sequences(scaled_data, M, N)

  # Random train/test split
  X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True
  )

  # Convert to PyTorch tensors
  X_train = torch.tensor(X_train, dtype=torch.float32)
  X_test = torch.tensor(X_test, dtype=torch.float32)
  y_train = torch.tensor(y_train, dtype=torch.float32)
  y_test = torch.tensor(y_test, dtype=torch.float32)

  return X_train, X_test, y_train, y_test, scaler

class Stock:
  def __init__(self, name: str):
    super().__init__()
    self.name = name

    X_train, X_test, y_train, y_test, scaler = prepare_data(name)
    self.train = TensorDataset(X_train, y_train)
    self.X_train = X_train
    self.y_train = y_train
    self.test = TensorDataset(X_test, y_test)
    self.X_test = X_test
    self.y_test = y_test
    self.scaler = scaler

def get_datasets(verbose=False):
  datasets = dict[str, Stock]()

  for name in STOCKS:
    if verbose:
      print(f"Processing {name}...")

    datasets[name] = stock = Stock(name)

    if verbose:
      print(f"{name} shapes:")
      print(f"  X_train: {stock.X_train.shape}")
      print(f"  X_test : {stock.X_test.shape}")
      print(f"  y_train: {stock.y_train.shape}")
      print(f"  y_test : {stock.y_test.shape}")
      print("-" * 40)

  return datasets

# Not saving in files this time probably better?
if __name__ == "__main__":
    datasets = get_datasets(verbose=True)
    print("Task 1 complete.")
