import yfinance as yf
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

"""
Intro to Machine Learning Assignment 4
Encompasses the solution to Task 1.
Students: Jackie Javier, Pranitha Achanta, Robert McDaniels
"""
STOCKS = ["AAPL", "MSFT", "GOOG", "AMZN"]
# Taken from the suggested dates
START_DATE = "2018-02-01"
END_DATE = "2021-01-31"

M = 60  # number of past days (input sequence length)
N = 1   # number of future days to predict

TEST_SIZE = 0.2
RANDOM_STATE = 42

main = False

# Download the data
def download_stock_data(stock):
  data = yf.download(stock, start=START_DATE, end=END_DATE)
  return data['Close'].values.reshape(-1, 1)

# Convert time series into supervised learning format
def create_sequences(data, M, N):
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
def prepare_data(stock):
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

def get_datasets():
  datasets = {}

  for stock in STOCKS:
    if(main):
      print(f"Processing {stock}...")
    X_train, X_test, y_train, y_test, scaler = prepare_data(stock)

    datasets[stock] = {
      "X_train": X_train,
      "X_test": X_test,
      "y_train": y_train,
      "y_test": y_test,
      "scaler": scaler
    }

    if(main):
      print(f"{stock} shapes:")
      print(f"  X_train: {X_train.shape}")
      print(f"  X_test : {X_test.shape}")
      print(f"  y_train: {y_train.shape}")
      print(f"  y_test : {y_test.shape}")
      print("-" * 40)

  return datasets

# Not saving in files this time probably better?
if __name__ == "__main__":
    main = True
    datasets = get_datasets()
    print("Task 1 complete.")
