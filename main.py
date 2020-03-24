from code.dataloader import Dataset
from code.modelV2 import AttentionNetwork
import torch
from torch import optim
import numpy as np
from tqdm import tqdm
import pickle as pkl

#set file location path
filepath = 'data_simulation'  #"C:\\Users\\laure\\OneDrive\\Desktop\\cnn-data"

#set settings
print_boo = True #set this to false if you don't want print statements
n_examples = 160 #set this to any N to get N examples, if you go above the max (148) then it will just give all examples #this is a seed for reproducibility
max_epochs = 1000
output = 'output/'

#load the data
DNA_dataset = Dataset(filepath, n_examples, print_boo) #create Dataset
model = AttentionNetwork()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_overall = []
    for epoch in range(1,max_epochs+1):
        print('Epoch {}/{}'.format(epoch,max_epochs+1))
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
            x = x.to(device)
            y = y.squeeze(dim=0).to(device) #
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


        test(idx_validation, epoch)



def test(idx_validation, epoch):
    print('Evaluating trained model')
    model.eval()
    test_loss = 0.
    test_error = 0.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i in idx_validation:
        x, y = DNA_dataset[i]
        x = x.to(device)
        y = y.squeeze(dim=0).to(device)
        print(x.shape, y.shape)

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
    train()





