from code.dataloader import Dataset
from code.model import AttentionNetwork
from code.stop import EarlyStopping
import torch
from torch.autograd import Variable
from torch import optim
import numpy as np
from tqdm import tqdm
import pickle as pkl
import argparse
import random


output = 'output/'

#defines arguments that can be passed to the model
parser = argparse.ArgumentParser(description='PyTorch DNA bags Model')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='if passed disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='int',
                    help = 'random seed (default:1)')
parser.add_argument('--dataN', type=int, default=160, metavar='int',
                    help = 'number of data points to load')
parser.add_argument('--lr', type=float, default=0.0001, metavar='float',
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=180, metavar='int',
                    help='max epochs')

parser.add_argument('--maxk', type=int, default=5,
                    help='Max_k for the pooling layer')

parser.add_argument('--L', type=int, default=500,
                    help='Hidden units of linear layer')

parser.add_argument('--D', type=int, default=128,
                    help='Hidden units attention')

parser.add_argument('--CNN', type=int, default=100,
                    help='hidden units CNN')

parser.add_argument('--fold', type=int, default=1,
                    help='Cross-validation shuffle fold')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


if args.cuda:
    torch.cuda.manual_seed_all(args.seed)
    print('GPU is working \n')


print('Loading the data \n')
# log_file.write('Loading the data \n')
DNA_dataset = Dataset(path='data_simulation', n_examples=args.dataN, print_boo=False) #max n_examples for sim_DAta 160

print('Initialize Model \n')
model = AttentionNetwork(args.L, args.D, args.maxk, args.CNN)

if args.cuda:
    model.cuda()


optimizer = optim.Adam(model.parameters(), lr=args.lr)
# stopping_training = EarlyStopping(patience=8)
stopping_test = EarlyStopping()

print('Selected Model settings: \n')
print('model:{}, lr:{}, L:{}, D:{}, CNN:{}, maxk:{}, fold:{} \n'.format(args.model, args.lr, args.L, args.D, args.CNN, args.maxk, args.fold))


def train():

    #get a shuffle of the data
    idx_train = DNA_dataset.get_random_shuffle(args.fold)
    print('idx train {}'.format(idx_train))
    validation = int(len(idx_train) * 0.1)
    idx_validation = idx_train[-validation:]
    idx_train = idx_train[:len(idx_train) - validation]

    loss_overall = []
    error_overall = []
    test_loss = []
    test_error = []


    for epoch in range(1,args.epochs+1):
        print('Epoch {}/{} \n'.format(epoch,args.epochs))
        train_loss = 0.
        train_error = 0.
        model.train()

        #shuffle the training data at every epoch
        random.seed(epoch)
        random.shuffle(idx_train)

        #train the model
        for i in tqdm(idx_train):
            x, y = DNA_dataset[i]
            y = y.squeeze(dim=0)
            if args.cuda:
                x, y = x.cuda(), y.cuda()

            x, y = Variable(x), Variable(y)

            #reset gradients
            optimizer.zero_grad()
            #calucalte loss
            loss, _ = model.calculate_objective(x.float(), y)
            train_loss += loss.item()
            error, _ = model.calculate_classification_error(x.float(), y)
            train_error += error
            # backward pass
            loss.backward()
            # step
            optimizer.step()

        # calculate loss and error for epoch
        train_loss /= len(idx_train)  # number of genes
        loss_overall.append(train_loss)
        train_error /= len(idx_train)
        error_overall.append(train_error)
        print('Epoch: {}, Loss: {:.4f}, train error: {:.4f} \n'.format(epoch, train_loss, train_error))

        if epoch%50==0:
            ckp_metrics = {'loss':loss_overall,'error':error_overall,
                           'loss_test':test_loss, 'error_test':test_error}

            checkpoint_name = str(args.model)+'_'+ str(args.lr) + '_' +\
                              str(args.L)+'_'+ str(args.CNN)+'_' + str(args.maxk) + \
                              '_fold_' + str(args.fold)

            pkl.dump(ckp_metrics, open('log/'+ checkpoint_name+'.pkl', 'wb'))

        t_loss, t_error =test(idx_validation, epoch)
        test_loss.append(t_loss)
        test_error.append(t_error)

        if stopping_test.is_better(t_loss):
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
            }

            ckp_name = str(args.model) + '_' + str(args.lr) + '_' + \
                              str(args.L) + '_' + str(args.CNN) + '_' + str(args.maxk) \
                                       + '_' + str(epoch) +'_fold_' + str(args.fold)+ '_best_vl'

            stopping_test.store_model(checkpoint, ckp_name)


        if stopping_test.num_bad_epochs>=stopping_test.patience:
            save_ckp(stopping_test.checkpoint, stopping_test.checkpoint_name, 'output/')

def save_ckp(state, checkpoint_name, ckp_dir):
    f_path = ckp_dir + checkpoint_name + '.pt'
    torch.save(state, f_path)


if __name__ == '__main__':
    train()






