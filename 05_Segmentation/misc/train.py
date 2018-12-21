import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


def separate_classes(tensor):
    ones = torch.ones(tensor.shape)
    inverted = ones - tensor
    return torch.cat((tensor, inverted), dim = 1)


def train_network(net, dl, dl_test=None, *, n_epoch = 100, schedule={0:0.001}):
    """
    Optimizes network with Adam, using BCELoss+DICELoss as criterion.

    schedule - list of tuples [(epoch, lr), ...];
    dl - train data loader;
    dl_test - test data loader;
    n_epoch - number of back-propagation steps;

    Returns tuple of two lists - train and test loss at every epoch of training.
    """
    full_loss_train = []
    full_loss_test = []

    net.cuda()
    criterion = nn.BCELoss().cuda()
    optimizer = optim.Adam(net.parameters(), lr=schedule[0], weight_decay=1e-5, amsgrad=True)

    for epoch in tqdm.tqdm_notebook(range(0, n_epoch)):
        epoch_loss = 0
        net.train(True)

        if epoch in schedule.keys():
            print('Learning rate: ', schedule[epoch])
            optimizer = optim.Adam(net.parameters(), lr=schedule[epoch])

        for iter_, (x, x_mask) in enumerate(dl):
            x = Variable(x).cuda()
            x_mask = separate_classes(x_mask)
            x_mask = Variable(x_mask).cuda()

            x_mask_guess = net(x)
            loss = criterion(x_mask_guess, x_mask)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(dl.dataset)
        full_loss_train.append(epoch_loss)

        net.train(False)
        if dl_test is not None:
            test_loss = 0
            for iter_, (x, x_mask) in enumerate(dl_test):
                x = Variable(x).cuda()
                x_mask = separate_classes(x_mask)
                x_mask = Variable(x_mask).cuda()
                x_mask_guess = net(x)
                loss = criterion(x_mask_guess, x_mask)
                test_loss += loss.item()
            test_loss = test_loss / len(dl_test.dataset)
            full_loss_test.append(test_loss)
            print('Epoch: %d | Train loss: %f, Test loss: %f' % (epoch, epoch_loss, test_loss))
        else:
            print('Epoch: %d | Train loss: %f' % (epoch, epoch_loss))
    return full_loss_train, full_loss_test
