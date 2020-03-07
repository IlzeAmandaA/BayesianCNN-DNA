import torch
from torch.utils import data
import pandas as pd

class Dataset(data.Dataset):
    def __init__(self, path):
        self.data = pd.read_csv(path)#read the data in for file path

    def __len__(self):
        return len(self.data) #pass the an int in which range index is created

    def __getitem__(self, index):
        #acess part of data based on a index (depending on data input format)
        X = self.data[1:,index]
        y = self.data[0, index]

        return X,y
