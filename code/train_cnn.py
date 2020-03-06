from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from CNN import CNN_model
import torch
import pickle
import matplotlib.pyplot as plt
from create_arbitrary_data import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

embedding_size = 4
sequence_length = 20
people = 100

def train(people, embedding_size, sequence_length):
	x_data = create_fake_x_data(people, embedding_size, sequence_length)
	y_data = create_fake_y_data(people)
	
	x_data = torch.from_numpy(x_data).float().to(device)
	y_data = torch.from_numpy(y_data).float().to(device)

	print(x_data.shape)
	print(y_data.shape)
	model = CNN_model(1,2).to(device)
	loss = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr = 1e-04, weight_decay=1e-6)
	#max_steps = 1000

	for i in range(people):
		x = x_data[i]
		y = y_data[i].long()

		out = model.forward(x.unsqueeze(0).unsqueeze(0))
		#print(out, ' ', y)
		loss_training = loss(out, y.unsqueeze(0))
		optimizer.zero_grad()
		loss_training.backward()
		optimizer.step()
		print("Current loss: ", loss_training.item())

train(people, embedding_size, sequence_length)