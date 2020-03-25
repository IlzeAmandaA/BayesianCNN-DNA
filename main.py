from code.dataloader import Dataset
from code.modelV2 import AttentionNetwork
import torch
from torch.autograd import Variable
from torch import optim
import numpy as np
from tqdm import tqdm
import pickle as pkl
import argparse

#defines arguments that can be passed to the model
parser = argparse.ArgumentParser(description='PyTorch DNA bags Model')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='if passed disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='int',
                    help = 'random seed (default:1)')
parser.add_argument('--dataN', type=int, default=160, metavar='int',
                    help = 'number of data points to load')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
    torch.cuda.manual_seed_all(args.seed)
    print('GPU is working')

print('Loading the data')
DNA_dataset = Dataset(path='data_simulation', n_examples=args.dataN, print_boo=True) #max n_examples for sim_DAta 160

print('Initialize Model')
model = AttentionNetwork()
if args.cuda:
    model.cuda()


optimizer = optim.Adam(model.parameters(), lr=0.01)


def train(max_epochs, output):
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_overall = []
    for epoch in range(1,max_epochs+1):
        print('Epoch {}/{}'.format(epoch,max_epochs))
        train_loss = 0.
        train_error = 0.
        model.train()

        idx_train = DNA_dataset.get_random_shuffle(epoch)  # get a random shuffle for cross validation, do this K times
        validation = int(len(idx_train) * 0.2)
        idx_validation = idx_train[-validation:]
        idx_train = idx_train[:len(idx_train) - validation]

        #train the model
        for i in tqdm(idx_train):
            x, y = DNA_dataset[i]
            y = y.squeeze(dim=0)
            if args.cuda:
                x, y = x.cuda(), y.cuda()

            x, y = Variable(x), Variable(y)
            # x = x.to(device)
            # y = y.squeeze(dim=0).to(device) #
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
        print('Epoch: {}, Loss: {:.4f}, train error: {:.4f}'.format(epoch, train_loss, train_error))
        if epoch%10==0:
            torch.save(model.state_dict(), output + 'model_epoch_' + str(epoch) + '_.pth')
            pkl.dump(loss_overall, open(output+'loss_train.pkl','wb'))


        test(idx_validation, epoch, output)



def test(idx_validation, epoch, output):
    print('Evaluating trained model')
    model.eval()
    test_loss = 0.
    test_error = 0.
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i in idx_validation:
        x, y = DNA_dataset[i]
        y=y.squeeze(dim=0)
        if args.cuda:
            x, y = x.cuda(), y.cuda()
        x, y = Variable(x), Variable(y)
        # x = x.to(device)
        # y = y.squeeze(dim=0).to(device)
        # print(x.shape, y.shape)

        loss, attention_weights = model.calculate_objective(x.float(), y)
        test_loss += loss.item()
        error, predicted_label = model.calculate_classification_error(x.float(), y)
        test_error += error


        bag_level = (y.cpu().data.numpy()[0], int(predicted_label.cpu().data.numpy()[0][0]))
        instance_level = list(np.round(attention_weights.cpu().data.numpy()[0], decimals=3))

        print('\nTrue Bag Label, Predicted Bag Label: {}\n'
                  'Attention Weights: {}'.format(bag_level, instance_level))

    test_error /= len(idx_validation)
    test_loss /= len(idx_validation)    # * x.shape[0]

    print('\nTest Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss, test_error))
    file_test = open(output+'test_lost.txt', 'a')
    file_test.write('Training epoch {} \n'.format(epoch))
    file_test.write('Test Set, Loss: {:.4f}, Test error: {:.4f} \n'.format(test_loss, test_error))
    file_test.close()


if __name__ == '__main__':
    train(max_epochs=1000, output='output/')





