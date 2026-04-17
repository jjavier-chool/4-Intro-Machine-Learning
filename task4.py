import torch
import torch.nn as nn

from task3 import train_eval

"""
Intro Machine Learning Assignment 4
Encompasses the solution to Task 2.
Code is adopted and modified from the class slides.
Students: Jackie Javier, Pranitha Achanta, Robert McDaniels
"""
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64

class GRUModel(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(GRUModel, self).__init__()

    self.rnn = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
    self.h2o = nn.Linear(hidden_size, output_size)
    # self.softmax = nn.LogSoftmax(dim=1) ignore for regression

  def forward(self, line_tensor):
    rnn_out,hidden = self.rnn(line_tensor)
    output = rnn_out[:, -1, :] # just last time step?
    output = self.h2o(output)
    return output

def main():
  torch.manual_seed(42)
  train_eval(GRUModel, HIDDEN_SIZE, LEARNING_RATE)

if __name__ == "__main__":
  main()