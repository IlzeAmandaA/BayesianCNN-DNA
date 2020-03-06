import torch
from torch import nn
import torch.nn.functional as F

from create_fake_data import *

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		
		self.maxpool1 = nn.MaxPool1d(kernel_size = 3, stride=3)
		self.dropout1 = nn.Dropout(p=0.5, inplace=False)
		self.relu1 = nn.ReLU(inplace = False)
		self.conv1 = [nn.Conv1d(in_channels = 4, out_channels = 100, kernel_size = 11),self.dropout1, self.relu1, self.maxpool1]
		self.all_layers = [*self.conv1]
		self.m = nn.Sequential(*self.all_layers)

	def forward(self, x):
		return self.m(x)

batch_size = 100 #people
features = 4 # features
seq_length = 2000 # amount of basepairs
kernel_size_conv1 = 11 #width of the first filter
kernel_size_maxpool1 = 3 #kernel size of the first dropout layer


#a = torch.randn(batch_size, features, seq_length)  
x_data = create_fake_x_data(batch_size, features, seq_length)

m = Model()

out = m.forward(x_data)

print("==========")
print("The architecture of the model: ", m)
print("==========")

print(f"Input size: {x_data.size()}")
print(f"Output size: {out.size()}")
