from model import AttentionNetwork
from dataloader import Dataset
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable

filepath = 'll'
dataset = Dataset(filepath)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
model = AttentionNetwork()
optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), weight_decay=1e-04)


def train():
    max_epochs = 1000
    for epoch in range(1, max_epochs+1):
        model.train() #set model to training mode
        train_loss = 0.
        train_error = 0.
        for batch_idx, (data, label) in enumerate(data_loader):
            bag_label = label[0]
            # if args.cuda:
            #     data, bag_label = data.cuda(), bag_label.cuda()
            data, bag_label = Variable(data), Variable(bag_label)
            # reset gradients
            optimizer.zero_grad()
            # calculate loss and metrics
            loss, _ = model.calculate_objective(data, bag_label)
            train_loss += loss.data[0]
            error, _ = model.calculate_classification_error(data, bag_label)
            train_error += error
            # backward pass
            loss.backward()
            # step
            optimizer.step()

        # calculate loss and error for epoch
        train_loss /= len(data_loader)
        train_error /= len(data_loader)

        print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss.cpu().numpy()[0], train_error))


if __name__ == '__main__':
    print('Started training the model')
    train()






