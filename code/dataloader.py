import torch
from torch.utils import data
import pandas as pd
import numpy as np

class Dataset(data.Dataset):
	def __init__(self, path):
		#self.data = pd.read_csv(path)#read the data in for file path
		self.data = self.importData(path)

	def __len__(self):
		return len(self.data) #pass the an int in which range index is created

	def __getitem__(self, index):
		#acess part of data based on a index (depending on data input format)
		#X = self.data[1:,index]
		#y = self.data[0, index]


		""" 
			To do: make it such that it accesses the data after applying appropiate padding
		"""
		print(self.data[0]['gene1'].shape)
		return 0

	def process_line(self, line):
		"""
			pre processes a given line to extract both the sequence and the correpsonding label
		"""
		pieces = line.split(",")
		seq = pieces[1]
		label = int(pieces[2])
		return seq, label

	def importData(self, filepath):
		"""
			Reads data given a filepath and returns the genes {gene1: "sequence", "gene2:" ..., ...} 
			with a corresponding label (1,0) <-- vector or scalar?
		"""
		f = open(filepath)
		genes= {}
		label = None
		for i,line in enumerate(f):
			line = line[:-1] #remove \n
			seq, lab = self.process_line(line)
			genes['gene'+str(i)] = self.DNA_to_onehot(seq)
			label=lab
		return genes, label

	def DNA_to_onehot(self, sequence):
		"""
			Converts a sequence to DNA one hot encoding
		"""
		look_up = {"A": 0, "G": 1, "T": 2, "C": 3}
		data = np.zeros((4, len(sequence))) # format of [4,len(sequence)]
		# print(data.shape)
		for index, letter in enumerate(sequence):
			data[look_up[letter]][index] = 1
		return torch.from_numpy(data).float()

	def add_padding(self):
		"""
			Adds appropiate padding where the size of the padding is determined by the largest bp sequnce (±98k)
		"""
		return 0

gene, label = Dataset("/Users/Laurence/Documents/GitHub/Bayesian-CNNs-and-DNA/data/id1.txt")

print(gene, label)

# To do:
# 	-Implement appropiate padding
# 	- See how the data holds on a CPU
#	- is gene1: seq the final representation of the data?
# 	- Question: load all data at once or go over it file by file, does this matter for the dataloader



