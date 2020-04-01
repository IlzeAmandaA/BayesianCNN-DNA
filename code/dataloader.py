from torch.utils import data
from code.read_data import *
import numpy as np

class Dataset(data.Dataset):
    def __init__(self, path, n_examples = 999, print_boo = True):
        self.X, self.y = set_up_data(path, n_examples, print_boo)
        print('X shape', self.X.shape)
        print('y shape', self.y.shape)


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


