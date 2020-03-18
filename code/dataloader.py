import torch
from torch.utils import data
import pandas as pd
from create_arbitrary_data import *
from read_data import *
import random
import numpy as np 

class Dataset(data.Dataset):
    def __init__(self, path, n_examples = 999, print_boo = True):
        self.X, self.y = set_up_data(path, n_examples, print_boo)

    def __len__(self):
        return len(self.X) #pass the an int in which range index is created

    def __getitem__(self, index):
        #acess part of data based on a index (depending on data input format)
        X = self.X[index,:,:,:]
        y = self.y[index,:]

        return X,y.unsqueeze(dim=1)

    def get_random_shuffle(self, seed = 0):
    	np.random.seed(seed)
    	idx = np.arange(len(self.X))
    	np.random.shuffle(idx)
    	return idx

filepath = "C:\\Users\\laure\\OneDrive\\Desktop\\cnn-data"

print_boo = True #set this to false if you don't want print statements
n_examples = 999 #set this to any N to get N examples, if you go above the max (148) then it will just give all examples
seed = 0 #this is a seed for reproducibility 

DNA_dataset = Dataset(filepath, n_examples, print_boo) #create Dataset
idx_shuffle = DNA_dataset.get_random_shuffle(seed) #get a random shuffle for cross validation, do this K times

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for i in idx_shuffle:
	x, y = DNA_dataset[i]
	x = x.to(device)
	y = y.to(device)
	print(x.shape, y.shape)
