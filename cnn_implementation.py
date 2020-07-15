# importing the libraries
import pandas as pd
import numpy as np

# for reading and displaying images
from skimage.io import imread
import matplotlib.pyplot as plt

# for creating validation set
from sklearn.model_selection import train_test_split

# for evaluating the model
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, \
    Dropout
from torch.optim import Adam, SGD


class Net(Module):
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(4 * 7 * 7, 10)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


def train(epoch):
    model.train()
    tr_loss = 0
    # getting the training set
    x_train, y_train = Variable(train_x), Variable(train_y)
    # getting the validation set
    x_val, y_val = Variable(val_x), Variable(val_y)
    # converting the data into GPU format
    if torch.cuda.is_available():
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        x_val = x_val.cuda()
        y_val = y_val.cuda()

    # clearing the Gradients of the model parameters
    optimizer.zero_grad()

    # prediction for training and validation set
    output_train = model(x_train)
    output_val = model(x_val)

    # computing the training and validation loss
    loss_train = criterion(output_train, y_train)
    loss_val = criterion(output_val, y_val)
    train_losses.append(loss_train)
    val_losses.append(loss_val)

    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()
    if epoch % 2 == 0:
        # printing the validation loss
        print('Epoch : ', epoch + 1, '\t', 'loss :', loss_val)


if __name__ == '__main__':
    global model, val_x, val_y, optimizer, criterion, n_epochs, train_losses, val_losses
    # loading dataset
    # train = pd.read_csv('train_LbELtWX/train.csv')
    # test = pd.read_csv('test_ScVgIM0/test.csv')
    train_df = pd.read_csv('classification/train.csv')
    # test = pd.read_csv('results/6.30.2020@12-9-6_5_cubic_5_reno_0_vegas/single_connection_stat_cubic_1.csv')

    sample_submission = pd.read_csv(
        'results/6.30.2020@12-9-6_5_cubic_5_reno_0_vegas/single_connection_stat_cubic_2.csv')

    train_df.head()

    # loading training images
    data_arr = []
    for df_name in tqdm(train_df['id']):
        # defining the image path
        csv_filename = 'results/6.30.2020@12-9-6_5_cubic_5_reno_0_vegas/single_connection_stat_%s.csv' % df_name
        df = pd.read_csv(csv_filename)
        df = df.drop(columns='Time')
        df = df.astype(np.float32)

        # appending the image into the list
        data_arr.append(df.to_numpy())

    # converting the list to numpy array
    train_x = np.array(data_arr)
    # defining the target
    train_y = train_df['label'].values
    train_x.shape

    # create validation set
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.1)
    (train_x.shape, train_y.shape), (val_x.shape, val_y.shape)

    # converting training images into torch format
    train_x = train_x.reshape(9, 1, 602, 8)
    train_x = torch.from_numpy(train_x)

    # converting the target into torch format
    train_y = train_y.astype(int);
    train_y = torch.from_numpy(train_y)

    # shape of training data
    train_x.shape, train_y.shape

    # converting validation images into torch format
    val_x = val_x.reshape(1, 1, 602, 8)
    val_x = torch.from_numpy(val_x)

    # converting the target into torch format
    val_y = val_y.astype(int);
    val_y = torch.from_numpy(val_y)

    # shape of validation data
    val_x.shape, val_y.shape

    # defining the model
    model = Net()
    # defining the optimizer
    optimizer = Adam(model.parameters(), lr=0.07)
    # defining the loss function
    criterion = CrossEntropyLoss()
    # checking if GPU is available
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    print(model)

    # defining the number of epochs
    n_epochs = 25
    # empty list to store training losses
    train_losses = []
    # empty list to store validation losses
    val_losses = []
    # training the model
    for epoch in range(n_epochs):
        train(epoch)

    # plotting the training and validation loss
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()
    plt.show()

    # prediction for training set
    with torch.no_grad():
        output = model(train_x.cuda())

    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)

    # accuracy on training set
    accuracy_score(train_y, predictions)
