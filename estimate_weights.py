from code.dataloader import Dataset
from code.modelV2 import AttentionNetwork
from code.modelDeep import AttentionNetworkDeep
import torch
import numpy as np
import pickle as pkl
from torch.autograd import Variable
from torch import optim
import argparse

def load_ckp(checkpoint_fpath, model):
    checkpoint = torch.load('output/best_models/' + checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    return model, checkpoint['epoch']

#defines arguments that can be passed to the model
parser = argparse.ArgumentParser(description='Weight Matrix Estiamtio ')
parser.add_argument('--filename', type=str, default='none',
                    help='Choose which model to evaluate')
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
assert args.filename != 'none' 'Filename missing'

args.cuda = not args.no_cuda and torch.cuda.is_available()

print('Initialize Model \n')
# log_file.write('Initialize Model \n')
if args.model == 'attention':
    print('Running attention model \n')
    model = AttentionNetwork(args.L, args.D, args.maxk, args.CNN)
elif args.model == 'attention_deep':
    model = AttentionNetworkDeep(args.L, args.D, args.maxk, args.CNN)

# optimizer = optim.Adam(model.parameters(), lr=args.lr)


model, epoch = load_ckp(args.filename + '.pt', model)
if args.cuda:
    model.cuda()


print('Loading the data \n')
# log_file.write('Loading the data \n')
DNA_dataset = Dataset(path='data_simulation', n_examples=args.dataN, print_boo=False) #max n_examples for sim_DAta 160




def validate():
    model.eval()
    test_loss = 0.
    test_error = 0.
    pickle_attention = []

    with torch.no_grad():
        for epoch in range(1,11):
            print('Epoch {}/{} \n'.format(epoch,10))
            idx_train = DNA_dataset.get_random_shuffle(epoch)  # get a random shuffle for cross validation, do this K times
            validation = int(len(idx_train) * 0.1)
            idx_validation = idx_train[-validation:]
            attention_weight_pos = []

            for i_n, i in enumerate(idx_validation):
                x, y = DNA_dataset[i]
                y = y.squeeze(dim=0)

                if args.cuda:
                    x, y = x.cuda(), y.cuda()
                x, y = Variable(x), Variable(y)

                loss, attention_weights = model.calculate_objective(x.float(), y)
                test_loss += loss.item()
                error, predicted_label = model.calculate_classification_error(x.float(), y)
                test_error += error


                if y==1:
                   attention_weight_pos.append(attention_weights)

            epoch_attention = torch.stack(attention_weight_pos, dim=1).squeeze(0)
            mean_epoch_attention = (epoch_attention.sum(0)/epoch_attention.shape[0]).numpy()

                #
                #
                # if i_n < 1:  # print info for 5 bags
                #     bag_level = (y.cpu().data.numpy()[0], int(predicted_label.cpu().data.numpy()[0][0]))
                #     instance_level = list(np.round(attention_weights.cpu().data.numpy()[0], decimals=3))

            #
            # print('\nTrue Bag Label, Predicted Bag Label: {}\n'
            #               'Attention Weights: {} \n'.format(bag_level, instance_level))
                # log_file.write('\nTrue Bag Label, Predicted Bag Label: {}\n'
                #           'Attention Weights: {} \n'.format(bag_level, instance_level))

            test_error /= len(idx_validation)
            test_loss /= len(idx_validation)  # * x.shape[0]
            pickle_attention.append(mean_epoch_attention)

            print('\nTest Set, Loss: {:.4f}, Test error: {:.4f} \n'.format(test_loss, test_error))
            # log_file.write('\nTest Set, Loss: {:.4f}, Test error: {:.4f} \n'.format(test_loss, test_error))
            file_test = open('output/'+'validation_' + args.filename + '.txt', 'a')
            file_test.write('Training epoch {} \n'.format(epoch))
            file_test.write('Test Set, Loss: {:.4f}, Test error: {:.4f} \n'.format(test_loss, test_error))
            file_test.write('Mean Attention Weights (label 1): \n')
            file_test.write(mean_epoch_attention)
            file_test.write('\n')
            file_test.close()


        pkl.dump(pickle_attention, open('output/'+'attentions_'+ args.filename+'.pkl', 'wb'))




if __name__ == '__main__':
    validate()