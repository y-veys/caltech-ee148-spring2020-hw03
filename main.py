from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

import os

'''
This code is adapted from two sources:
(i) The official PyTorch MNIST example (https://github.com/pytorch/examples/blob/master/mnist/main.py)
(ii) Starter code from Yisong Yue's CS 155 Course (http://www.yisongyue.com/courses/cs155/2020_winter/)
'''

class fcNet(nn.Module):
    '''
    Design your model with fully connected layers (convolutional layers are not
    allowed here). Initial model is designed to have a poor performance. These
    are the sample units you can try:
        Linear, Dropout, activation layers (ReLU, softmax)
    '''
    def __init__(self):
        # Define the units that you will use in your model
        # Note that this has nothing to do with the order in which operations
        # are applied - that is defined in the forward function below.
        super(fcNet, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=20)
        self.fc2 = nn.Linear(20, 10)
        self.dropout1 = nn.Dropout(p=0.5)

    def forward(self, x):
        # Define the sequence of operations your model will apply to an input x
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = F.relu(x)

        output = F.log_softmax(x, dim=1)
        return output


class ConvNet(nn.Module):
    '''
    Design your model with convolutional layers.
    '''
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1)
        self.conv2 = nn.Conv2d(8, 8, 3, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(200, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output


class Net(nn.Module):
    '''
    Build the best MNIST classifier.
    '''
    def __init__(self):
        super(Net, self).__init__()

        self.model = nn.Sequential(
                nn.Conv2d(1, 9, kernel_size=(3,3)),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(p=0.2),

                nn.Conv2d(9, 16, kernel_size=(3,3)),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(p=0.2),

                nn.Flatten(),
                nn.Linear(400, 64),
                nn.ReLU(),
                nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)

        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    model.train()   # Set the model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()               # Clear the gradient
        output = model(data)                # Make predictions
        loss = F.nll_loss(output, target)   # Compute loss
        loss.backward()                     # Gradient computation
        optimizer.step()                    # Perform a single optimization step
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()    # Set the model to inference mode
    test_loss = 0
    correct = 0
    test_num = 0
    inc_predictions = []
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item() 
            test_num += len(data)
            
            # inc_indices = np.where(pred.eq(target.view_as(pred))==False)[0]
            # for ind in inc_indices:
            #     inc_data = data[ind].detach().numpy().squeeze()
            #     inc_pred = pred[ind].squeeze().numpy()

            #     inc_predictions.append((inc_data, inc_pred))

            # plt.imshow(confusion_matrix(target, pred)/10000)
            # plt.title("Normalized Confusion Matrix")
            # plt.xlabel("Predicted Value")
            # plt.ylabel("True Value")
            # plt.show()

    test_loss /= test_num

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_num,
        100. * correct / test_num))

    #show_mistakes(inc_predictions)

    return test_loss

def show_kernels(model):

    fig, axes = plt.subplots(3,3)
    fig.suptitle("Kernels of First Layer")

    i = 0 
    for ax in axes.flat:
            kernel = model.model[0].weight.detach().numpy().squeeze()[i]
            im = ax.imshow(kernel)

            i += 1

    plt.colorbar(im, ax=axes.ravel().tolist())
    plt.show()

def show_mistakes(inc_predictions):

    fig, axes = plt.subplots(3,3)
    fig.suptitle("Nine Examples Classifier Got Incorrect")

    i = 0 
    for ax in axes.flat:
            im = ax.imshow(inc_predictions[i][0], cmap='gray')
            title = 'Predicted Value: ' + str(inc_predictions[i][1])
            ax.set_title(title, fontsize=9)

            i += 1

    plt.show()


def feature_vector(model, device, test_loader):
    model.eval()    # Set the model to inference mode
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model.model[0](data)

            for i in range(1,10):
                output = model.model[i](output)

            output = output.detach().numpy().squeeze()
            X_2d = TSNE(n_components=2).fit_transform(output)

        plt.scatter(X_2d[:,0], X_2d[:,1], c=target, cmap=plt.cm.get_cmap("jet", 10),s=1)
        plt.colorbar(ticks=range(10))
        plt.clim(-0.5, 9.5)
        plt.title('High Dimensional Feature Embedding using tSNE')
        plt.show()

def find_nearest_vectors(model, device, test_loader):
    outputs = []
    model.eval()    # Set the model to inference mode
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model.model[0](data)

            for i in range(1,10):
                output = model.model[i](output)

            output = output.detach().numpy().squeeze()
            outputs.append(output)

    indices = random.sample(range(10000), 8)
    images = []

    for i in indices: 
        image = data[i].detach().numpy().squeeze()
        true = target[i].squeeze().numpy()
        vector = outputs[0][i]

        distances = []
        for output in outputs[0]: 
            distances.append(np.linalg.norm(vector-output))

        min_8_indices = np.argpartition(np.array(distances),8)[:8]
        for j in min_8_indices: 
            images.append(data[j].detach().numpy().squeeze())


    fig, axes = plt.subplots(8,8)
    fig.suptitle("Images with Similar Feature Vectors")

    i = 0 
    for ax in axes.flat:
            im = ax.imshow(images[i], cmap='gray')
            i += 1

    plt.show()


def main():
    # Training settings
    # Use the command line to modify the default settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--step', type=int, default=1, metavar='N',
                        help='number of epochs between learning rate reductions (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate your model on the official test set')
    parser.add_argument('--load-model', type=str,
                        help='model file path')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    np.random.seed(1)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Evaluate on the official test set
    if args.evaluate:
        assert os.path.exists(args.load_model)

        # Set the test model
        model =  Net().to(device)
        model.load_state_dict(torch.load(args.load_model))

        test_dataset = datasets.MNIST('../data', train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        test(model, device, test_loader)

        # Extra Visualization - Do with batch size = 10000 for ease 
        # show_kernels(model)
        # feature_vector(model, device, test_loader)
        # find_nearest_vectors(model, device, test_loader)

        return

    # Pytorch has default MNIST dataloader which loads data at each iteration
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                    #transforms.RandomAffine(degrees=10,translate=(0.03,0.03),
                    #                             scale=(0.75,1.5))
                ]))


    # You can assign indices for training/validation or use a random subset for
    # training by using SubsetRandomSampler. Right now the train and validation
    # sets are built from the same indices - this is bad! Change it so that
    # the training and validation sets are disjoint and have the correct relative sizes.
    targets = train_dataset.targets.numpy()

    subset_indices_train = []
    subset_indices_valid = []

    for i in range(10):
        # Location of classes, shuffled 
        ind_locs = np.ndarray.tolist(np.random.permutation(np.where(targets==i)[0]))

        # Sample 85% of the locations for training
        split_index = round(0.85*len(ind_locs))

        # Append to the training and validation sets 
        subset_indices_train += ind_locs[0:split_index]
        subset_indices_valid += ind_locs[split_index:]

    # subset_indices_train = np.random.permutation(subset_indices_train)
    # split_index = round(len(subset_indices_train)/16)
    # subset_indices_train = subset_indices_train[0:split_index]

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=SubsetRandomSampler(subset_indices_train)
    )
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.test_batch_size,
        sampler=SubsetRandomSampler(subset_indices_valid)
    )

    # Load your model [fcNet, ConvNet, Net]
    model = Net().to(device)

    # Try different optimzers here [Adam, SGD, RMSprop]
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # Set your learning rate scheduler
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    train_loss = []
    val_loss = []
    # Training loop
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        train_loss.append(test(model, device, train_loader))
        val_loss.append(test(model, device, val_loader))
        scheduler.step()    # learning rate scheduler

        # You may optionally save your model at each epoch here

    if args.save_model:
       torch.save(model.state_dict(), "model_best_9_kernels.pt")

    #print(train_loss)
    #print(val_loss)
    epoch_list = np.linspace(1,args.epochs,args.epochs)
    plt.figure()
    plt.plot(epoch_list, train_loss)
    plt.plot(epoch_list, val_loss)
    plt.title("Training vs. Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.legend(["Training Loss","Validation Loss"])
    plt.show()


if __name__ == '__main__':
    main()
