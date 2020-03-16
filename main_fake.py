from code.modelV2 import AttentionNetwork
from code.dataloader import Dataset
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
import os
import numpy as np
import torch
from tqdm import tqdm
import random

# filepath = 'll'
# dataset = Dataset(filepath)
# data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
model = AttentionNetwork()
optimizer = optim.Adam(model.parameters(), lr=0.001)
look_up = {"A": 0, "G": 1, "T": 2, "C": 3}

def DNA_to_onehot(sequence):
    data = np.zeros((len(sequence), 4))
    # print(data.shape)
    for index, letter in enumerate(sequence):
        data[index][look_up[letter]] = 1
    return torch.from_numpy(data).float()

def process_line(line):
    pieces = line.split(",")
    seq = pieces[1]
    label = int(pieces[2])
    return seq, label


def importData(filepath):
    f = open(filepath)
    genes= {}
    label = None
    for i,line in enumerate(f):
        line = line[:-1] #remove \n
        seq, lab = process_line(line)
        genes['gene'+str(i)] = DNA_to_onehot(seq)
        label=lab

    return genes, label



def train():
    max_epochs = 1000
    sample_label = torch.distributions.Bernoulli(torch.tensor([0.5]))
    for epoch in range(1, max_epochs+1):
        print('Epoch {}/{}'.format(epoch,max_epochs))
        model.train() #set model to training mode
        train_loss = 0.
        train_error = 0.
        for i in tqdm(range(100)):
            genes = torch.randint(0,2,(20,4,2000))
            label = sample_label.sample()
            print(label.shape)
            # reset gradients
            optimizer.zero_grad()
            # calculate loss and metrics
            loss, _ = model.calculate_objective(genes.float(), label)
            train_loss += loss.item()
            error, _ = model.calculate_classification_error(genes.float(), label)
            # train_error += error
            # backward pass
            loss.backward()
            # step
            optimizer.step()


        # calculate loss and error for epoch
        train_loss /= 100*20
        # train_error /= len(data_loader)
        print('Epoch: {}, Loss: {:.4f}'.format(epoch, train_loss))


if __name__ == '__main__':
    print('Started training the model')
    train()






