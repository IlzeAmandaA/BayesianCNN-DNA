"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class CNN_model(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """
  class Flatten(nn.Module):
    def __init__(self):
        super(CNN_model.Flatten, self).__init__()

    def forward(self, x):
      """
        Makes sure the data is in the correct shape
      """
      x = x.view(x.size(0), -1)
      return x

  def __init__(self, n_channels, n_classes):
    """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem

    """

    super(CNN_model, self).__init__()
    kernel_size = 3

    maxpool1 = nn.MaxPool2d(kernel_size = kernel_size, stride = 2, padding = 1)
    batchnorm1 = nn.BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    conv1 = [nn.Conv2d(in_channels=n_channels, out_channels=20, kernel_size = kernel_size, stride = 1, padding = 1), maxpool1, batchnorm1, nn.ReLU()]


    maxpool2 = nn.MaxPool2d(kernel_size = kernel_size, stride = 2, padding = 1)
    batchnorm2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    conv2 = [nn.Conv2d(20, 128, kernel_size = kernel_size, stride = 1, padding = 1), maxpool2, batchnorm2, nn.ReLU()]

    linear = torch.nn.Linear(640, n_classes)

    all_layers = [*conv1, *conv2, CNN_model.Flatten(), linear]

    self.model = nn.Sequential(*all_layers)
    print(self.model)

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    """
    out = self.model(x)


    return out