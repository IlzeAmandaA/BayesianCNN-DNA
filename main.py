from code.dataloader import Dataset
from code.modelV2 import AttentionNetwork
from code.modelDeep import AttentionNetworkDeep
from code.compareLoss import EarlyStopping
import torch
from torch.autograd import Variable
from torch import optim
import numpy as np
from tqdm import tqdm
import pickle as pkl
import argparse


output = 'output/'

#defines arguments that can be passed to the model
parser = argparse.ArgumentParser(description='PyTorch DNA bags Model')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='if passed disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='int',
                    help = 'random seed (default:1)')
parser.add_argument('--dataN', type=int, default=160, metavar='int',
                    help = 'number of data points to load')
parser.add_argument('--lr', type=float, default=0.001, metavar='float',
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=200, metavar='int',
                    help='max epochs')

parser.add_argument('--model', type=str, default='attention',
                    help='Choose which model to run (default attention)')

parser.add_argument('--maxk', type=int, default=5,
                    help='Max_k for the pooling layer')

parser.add_argument('--L', type=int, default=1000,
                    help='Hidden units of linear layer')

parser.add_argument('--D', type=int, default=500,
                    help='Hidden units attention')

parser.add_argument('--CNN', type=int, default=50,
                    help='hidden units CNN')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


if args.cuda:
    torch.cuda.manual_seed_all(args.seed)
    print('GPU is working \n')


print('Loading the data \n')
# log_file.write('Loading the data \n')
DNA_dataset = Dataset(path='data_simulation', n_examples=args.dataN, print_boo=False) #max n_examples for sim_DAta 160

print('Initialize Model \n')
# log_file.write('Initialize Model \n')
if args.model == 'attention':
    print('Running attention model \n')
    model = AttentionNetwork(args.L, args.D, args.maxk, args.CNN)
elif args.model == 'attention_deep':
    model = AttentionNetworkDeep(args.L, args.D, args.maxk, args.CNN)


if args.cuda:
    model.cuda()


optimizer = optim.Adam(model.parameters(), lr=args.lr)
stopping_training = EarlyStopping(patience=8)
stopping_test = EarlyStopping(patience=8)


def train():
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_overall = []
    error_overall = []
    test_loss = []
    test_error = []

    for epoch in range(1,args.epochs+1):
        print('Epoch {}/{} \n'.format(epoch,args.epochs))
        # log_file.write('Epoch {}/{} \n'.format(epoch,args.epochs))
        train_loss = 0.
        train_error = 0.
        model.train()

        idx_train = DNA_dataset.get_random_shuffle(epoch)  # get a random shuffle for cross validation, do this K times
        validation = int(len(idx_train) * 0.1)
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
        error_overall.append(train_error)
        print('Epoch: {}, Loss: {:.4f}, train error: {:.4f} \n'.format(epoch, train_loss, train_error))
        # log_file.write('Epoch: {}, Loss: {:.4f}, train error: {:.4f} \n'.format(epoch, train_loss, train_error))
        if epoch%20==0:
            checkpoint = {
                'epoch' : epoch,
                'state_dict' : model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }
            ckp_metrics = {'loss':loss_overall,'error':error_overall,
                           'loss_test':test_loss, 'error_test':test_error}

            checkpoint_name = str(args.model)+'_'+ str(args.lr) + '_' +\
                              str(args.L)+'_'+ str(args.CNN)+'_' + str(args.maxk)

            #save checkpoint of model trained settings
            save_ckp(checkpoint, checkpoint_name +'_'+str(epoch), 'log/')
            #save metric values
            pkl.dump(ckp_metrics, open('log/'+ checkpoint_name+'.pkl', 'wb'))

        if stopping_training.is_better(train_loss):
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            ckp_name = str(args.model) + '_' + str(args.lr) + '_' + \
                              str(args.L) + '_' + str(args.CNN) + '_' + str(args.maxk) \
                                       + '_' + str(epoch) + '_best_train'
            stopping_training.store_model(checkpoint, ckp_name)


        if stopping_training.num_bad_epochs>=stopping_training.patience:
            save_ckp(stopping_training.checkpoint, stopping_training.checkpoint_name, 'output/')

            # torch.save(model.state_dict(), output + 'model_epoch_' + str(epoch) + '_' + str(args.lr)+'.pth')
            # pkl.dump(loss_overall, open(output+'loss_train.pkl','wb'))
            #

        t_loss, t_error =test(idx_validation, epoch)
        test_loss.append(t_loss)
        test_error.append(t_error)

        if stopping_test.is_better(t_loss):
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            ckp_name = str(args.model) + '_' + str(args.lr) + '_' + \
                              str(args.L) + '_' + str(args.CNN) + '_' + str(args.maxk) \
                                       + '_' + str(epoch) + '_best_test'
            stopping_test.store_model(checkpoint, ckp_name)


        if stopping_test.num_bad_epochs>=stopping_test.patience:
            save_ckp(stopping_test.checkpoint, stopping_test.checkpoint_name, 'output/')





def test(idx_validation, epoch):
    print('Evaluating trained model')
    model.eval()
    test_loss = 0.
    test_error = 0.
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i_n,i in enumerate(idx_validation):
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

        if i_n<1: #print info for 5 bags
            bag_level = (y.cpu().data.numpy()[0], int(predicted_label.cpu().data.numpy()[0][0]))
            instance_level = list(np.round(attention_weights.cpu().data.numpy()[0], decimals=3))

            print('\nTrue Bag Label, Predicted Bag Label: {}\n'
                  'Attention Weights: {} \n'.format(bag_level, instance_level))
        # log_file.write('\nTrue Bag Label, Predicted Bag Label: {}\n'
        #           'Attention Weights: {} \n'.format(bag_level, instance_level))

    test_error /= len(idx_validation)
    test_loss /= len(idx_validation)    # * x.shape[0]

    print('\nTest Set, Loss: {:.4f}, Test error: {:.4f} \n'.format(test_loss, test_error))
    # log_file.write('\nTest Set, Loss: {:.4f}, Test error: {:.4f} \n'.format(test_loss, test_error))
    file_test = open(output+'test_lost_'+
                     str(args.model) + '_' + str(args.lr) + '_' +
                     str(args.L) + '_' + str(args.CNN) + '_' + str(args.maxk) +'.txt', 'a')
    file_test.write('Training epoch {} \n'.format(epoch))
    file_test.write('Test Set, Loss: {:.4f}, Test error: {:.4f} \n'.format(test_loss, test_error))
    file_test.close()
    return test_loss, test_error

def save_ckp(state, checkpoint_name, ckp_dir):
    f_path = ckp_dir + checkpoint_name + '.pt'
    torch.save(state, f_path)


if __name__ == '__main__':
    train()






