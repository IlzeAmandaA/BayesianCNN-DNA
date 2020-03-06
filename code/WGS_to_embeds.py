"""
	Given a string of "AGTC" the class Converter will map this sequence to a [4,sequence_length] matrix
"""
import numpy as np 
import torch

class Converter():
	def __init__(self, sequence):
		self.look_up = {"A": 0, "G": 1, "T": 2, "C": 3}
		self.embedding_size = 4
		self.sequence_length = len(sequence)
		self.sequence = sequence
		self.all_data = []
		self.add_sequence()

	def add_sequence(self):
		self.data = np.zeros((self.embedding_size, self.sequence_length))
		for index, letter in enumerate(self.sequence):
			self.data[index][self.look_up[letter]] = 1

		self.all_data.append(self.data)
		return torch.from_numpy(self.data).float()

	def return_all_data(self):
		return torch.from_numpy(np.asarray(self.all_data)).float()

	def set_sequence(self, sequence):
		self.sequence = sequence

""" 
	The flow of this script works as follows:
		- Obtain a sequence in the form of sequence = "AGTC"
		- Create a Converter object using Converter(sequence)
		- use "object_name.add_sequence(sequence)" to add a sequence to the data
"""

sequence = "AGTC"
data_converter = Converter(sequence)

data_converter.set_sequence("CTGA")
data_converter.add_sequence()

all_data = data_converter.return_all_data()

print(all_data)
print(all_data.shape) #form of [batch_size, embedding_size, sequence_length]