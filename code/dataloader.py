import torch
from torch.utils import data
import pandas as pd
from create_arbitrary_data import *

class Dataset(data.Dataset):
    def __init__(self, path):
        #self.data = pd.read_csv(path)#read the data in for file path
        batch_size = 100
        features = 4
        seq_length = 2000
        self.data = create_fake_x_data(batch_size, features, seq_length)
        self.y_data = create_fake_y_data(batch_size)

    def __len__(self):
        return len(self.data) #pass the an int in which range index is created

    def __getitem__(self, index):
        #acess part of data based on a index (depending on data input format)

        X = self.data[index:,:]
        y = self.y_data[index:]
        return X,y
