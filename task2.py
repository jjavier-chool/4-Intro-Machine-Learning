"""
Intro Machine Learning Assignment 4
Encompasses the solution to Task 2.
Code is adopted and modified from the class slides.
Students: Jackie Javier, Pranitha Achanta, Robert McDaniels
"""
import torch
import torch.nn as nn

class RNNModel(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, num_layers):
    super(RNNModel, self).__init__()

    self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True)
    self.h2o = nn.Linear(hidden_size, output_size)
    # self.softmax = nn.LogSoftmax(dim=1) ignore for regression

  def forward(self, line_tensor):
    rnn_out,hidden = self.rnn(line_tensor)
    output = rnn_out[:, -1, :] # just last time step?
    output = self.h2o(output)
    return output

if __name__ == "__main__":
  model = RNNModel(1, 64, 1)

  x = torch.randn(32, 60, 1)
  y = model(x)

  print("Output shape:", y.shape)  # Expected: (32, 1)
