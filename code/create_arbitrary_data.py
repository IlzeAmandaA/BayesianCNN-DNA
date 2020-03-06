"""" 
	This module will create arbritary "DNA" data in the form of [n_people, embedding_size, sequence_length]
"""
import torch
import numpy as np

embedding_size = 4
sequence_length = 2000
people = 100

def create_fake_batch(embedding_size, sequence_length):
	"""
		Creates a numpy matrix with size (embedding_size, sequence_length) where every column has 3 zeros and 1 one
	"""
	one_batch = np.zeros((embedding_size, sequence_length))
	for i in range(sequence_length):
		random_number = np.random.random_integers(0,3)
		one_batch[random_number][i] = 1
	return one_batch.astype(np.float64)

def create_fake_x_data(people, embedding_size, sequence_length):
	"""
		Creates a numpy matrix in the shape of (people, embedding_size, sequence_length)
	"""
	all_data = np.empty((people, embedding_size, sequence_length))
	for i in range(people):
		all_data[i] = create_fake_batch(embedding_size, sequence_length)
	return torch.from_numpy(all_data).float()

def create_fake_y_data(people):
	"""
		Generates n_people random labels of either 1 or 0
	"""
	return np.random.binomial(1, 0.5, people)

"""
fake_data = create_fake_batch(embedding_size, sequence_length)


for i in range(people):
	batch = torch.from_numpy(fake_data[i]).float().to(device)
	print(batch)
"""
