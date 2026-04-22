"""
Intro Machine Learning Assignment 4
Encompasses the solution to Task 5.
Code is adopted and modified from the class slides.
Students: Jackie Javier, Pranitha Achanta, Robert McDaniels
"""
import torch
import torch.nn as nn

from task3 import train_eval

LEARNING_RATE = 0.0005
HIDDEN_SIZE = 64
NUM_LAYERS = 1

class LSTM(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, num_layers=1):
    super(LSTM, self).__init__()

    self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
    self.h2o = nn.Linear(hidden_size, output_size)
    # self.softmax = nn.LogSoftmax(dim=1) ignore for regression

  def forward(self, line_tensor):
    rnn_out,(hidden, c) = self.rnn(line_tensor)
    #output = rnn_out[:, -1, :] # just last time step?
    output = self.h2o(hidden[-1])
    return output

def test(verbose=True):
  train_eval(LSTM, HIDDEN_SIZE, NUM_LAYERS, LEARNING_RATE, verbose=verbose)

def main():
  torch.manual_seed(42)
  train_eval(LSTM, HIDDEN_SIZE, NUM_LAYERS, LEARNING_RATE)

if __name__ == "__main__":
  main()
