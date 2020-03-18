from code.model import AttentionNetwork
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



def train(filepaths):
    max_epochs = 1000
    for epoch in tqdm(range(1, max_epochs+1)):
        model.train() #set model to training mode
        train_loss = 0.
        train_error = 0.
        random.shuffle(filepaths) #shuffle the filepahts aka the order of which participant is loaded when
        for batch_idx, file in enumerate(filepaths):
            gene_dict, label= importData(file)

          #  bag_label = label[0]
            # if args.cuda:
            #     data, bag_label = data.cuda(), bag_label.cuda()
          #  data, bag_label = Variable(data), Variable(bag_label)

            # reset gradients
            optimizer.zero_grad()
            # calculate loss and metrics
            loss, _ = model.calculate_objective(gene_dict, float(label))
            train_loss += loss.item()
            # error, _ = model.calculate_classification_error(data, bag_label)
            # train_error += error
            # backward pass
            loss.backward()
            # step
            optimizer.step()


        # calculate loss and error for epoch
        train_loss /= len(filepaths)*10
        # train_error /= len(data_loader)
        print('Epoch: {}, Loss: {:.4f}'.format(epoch, train_loss))


if __name__ == '__main__':
    path = './data/'
    files = [path+name for name in os.listdir(path)]
    print('Started training the model')
    train(files)






