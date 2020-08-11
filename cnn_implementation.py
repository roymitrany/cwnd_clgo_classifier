# importing the libraries
import pandas as pd
import numpy as np

# for reading and displaying graphs
# from skimage.io import imread
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

import glob
import os

NUM_OF_CLASSIFICATION_PARAMETERS = 9 # 7
NUM_OF_TIME_SAMPLES = 301 # 602
NUM_OF_CONGESTION_CONTROL_LABELING = 3
NUM_OF_CONV_FILTERS = 10
NUM_OF_TRAIN_DATAFRAMES = 7 # 9
NUM_OF_TEST_DATAFRAMES = 7

class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        # conv2d: output size = image_size - filter_size / stride + 1. usually stride = 1, filter [x,x], and padding = (f-1)/2.
        # maxpool2d: output size = image_size - filter_size.
        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, NUM_OF_CONV_FILTERS, kernel_size=(3, NUM_OF_CLASSIFICATION_PARAMETERS), stride=1, padding=(1, 0)),
            # channels: 1, filters: 10.
            BatchNorm2d(NUM_OF_CONV_FILTERS),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=(2, 1), stride=1),
            # Defining another 2D convolution layer
            Conv2d(NUM_OF_CONV_FILTERS, NUM_OF_CONV_FILTERS, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            BatchNorm2d(NUM_OF_CONV_FILTERS),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=(2, 1), stride=1),
        )

        self.linear_layers = Sequential(
            Linear(NUM_OF_CONV_FILTERS * (NUM_OF_TIME_SAMPLES - 2) * 1, NUM_OF_CONGESTION_CONTROL_LABELING + 1) # an errorbecause the labels must be 0 indexed. So, for example, if you have 20 classes, and the labels are 1th indexed, the 20th label would be 20, so cur_target < n_classes assert would fail. If it’s 0th indexed, the 20th label is 19, so cur_target < n_classes assert passes.
            # input features: 10 channels * number of rows * number of columns, output features: number of labels = 2.
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)  # flattening?
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
    output_train = model(x_train.type('torch.FloatTensor'))
    output_val = model(x_val.type('torch.FloatTensor'))

    # computing the training and validation loss
    loss_train = criterion(output_train, y_train.type('torch.LongTensor'))  # Long instead of float (was float and changed to long- now works).
    loss_val = criterion(output_val, y_val.type('torch.LongTensor'))
    train_losses.append(loss_train)
    val_losses.append(loss_val)

    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()
    if epoch % 2 == 0:
        # printing the validation loss
        print('Epoch : ', epoch + 1, '\t', 'loss :', loss_val)


def preprocessing_probabilistic(dataframe_arr):
    dataframe_arr_concatenated = pd.concat([dataframe for dataframe in dataframe_arr], axis=0)
    dataframe_mean = dataframe_arr_concatenated.mean()
    dataframe_std = dataframe_arr_concatenated.std()
    """
    dataframe_mean = [dataframe.mean().to_frame().T for dataframe in dataframe_arr]
    dataframe_std = [dataframe.std().to_frame().T for dataframe in dataframe_arr]
    return dataframe_mean, dataframe_std
    """
    # return [((dataframe - dataframe_mean) / np.sqrt(dataframe_mean)) for dataframe in dataframe_arr]
    return [((dataframe - dataframe_mean) / np.sqrt(dataframe_std)).fillna(0) for dataframe in dataframe_arr]



def preprocessing_absolute(dataframe_arr):
    return


if __name__ == '__main__':
    global model, val_x, val_y, optimizer, criterion, n_epochs, train_losses, val_losses
    # defining the dataframe path
    # path = r'C:\Users\deanc\PycharmProjects\Congestion_Control_Classifier\results\cnn_data'
    path = r'C:\Users\deanc\PycharmProjects\Congestion_Control_Classifier\train_files\8.4.2020@15-14-44_2_cubic_2_reno_3_bbr'
    all_files = glob.glob(os.path.join(path, "*.csv"))
    # loading dataset
    trainning_labeling = pd.read_csv(os.path.join(path, "train.csv"))
    trainning_labeling.head()
    # loading training dataframes
    dataframe_arr = []
    # all_files = glob.glob(path + "/*.csv")
    for csv_filename in all_files:
        # csv_file = pd.read_csv(csv_filename, index_col=None, header=0)
        csv_file = pd.read_csv(csv_filename)
        if csv_file.shape[0] < NUM_OF_TIME_SAMPLES:
            continue
        # df = df.drop(columns=['Time','Send Time Gap','Out Throughput', 'Connection Num of Drops', 'Total Bytes in Queue', 'Num of Packets', 'Num of Drops'])
        # appending the image into the list:
        # csv_file = csv_file.drop(columns=['Time', 'Send Time Gap'])
        # data_arr.append(csv_file.to_numpy())
        csv_file.dropna(inplace= True, how='all')
        dataframe_arr.append(csv_file)

    # converting the list to numpy array after pre- processing
    dataframe_arr = preprocessing_probabilistic(dataframe_arr)
    dataframe_arr = [dataframe.to_numpy() for dataframe in dataframe_arr]

    train_x = np.array(dataframe_arr)
    # defining the target
    train_y = np.array(trainning_labeling['label'].values)

    # create validation set
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.1)

    # converting training dataframes into torch format
    train_x = train_x.reshape(len(train_x), 1, NUM_OF_TIME_SAMPLES, NUM_OF_CLASSIFICATION_PARAMETERS)
    train_x = torch.from_numpy(train_x)

    # converting the target into torch format
    train_y = train_y.astype(float)
    train_y = torch.from_numpy(train_y)

    # converting validation dataframes into torch format
    val_x = val_x.reshape(NUM_OF_TRAIN_DATAFRAMES - len(train_y), 1, NUM_OF_TIME_SAMPLES, NUM_OF_CLASSIFICATION_PARAMETERS)
    val_x = torch.from_numpy(val_x)

    # converting the target into torch format
    val_y = val_y.astype(float)
    val_y = torch.from_numpy(val_y)

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
        output = model(train_x.type('torch.FloatTensor'))

    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)

    # accuracy on training set
    print(accuracy_score(train_y, predictions))

    # prediction for validation set
    with torch.no_grad():
        output = model(val_x.type('torch.FloatTensor'))

    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)

    # accuracy on validation set
    print(accuracy_score(val_y, predictions))

    # the test set:
    path = r'C:\Users\deanc\PycharmProjects\Congestion_Control_Classifier\test_files'
    all_files = glob.glob(os.path.join(path, "*.csv"))
    dataframe_arr = []
    for csv_filename in all_files:
        csv_file = pd.read_csv(csv_filename)
        if csv_file.shape[0] < NUM_OF_TIME_SAMPLES:
            continue
        # csv_file = csv_file.drop(columns=['timestamp', 'Send Time Gap'])
        # csv_file = csv_file.dropna()
        csv_file.dropna(inplace= True, how='all')
        # csv_file.replace()
        dataframe_arr.append(csv_file)
    test_x = preprocessing_probabilistic(dataframe_arr)
    test_x = [dataframe.to_numpy() for dataframe in test_x]
    test_x = np.array(test_x)
    test_x = test_x.reshape(NUM_OF_TEST_DATAFRAMES, 1, NUM_OF_TIME_SAMPLES, NUM_OF_CLASSIFICATION_PARAMETERS)
    test_x = torch.from_numpy(test_x)
    with torch.no_grad():
        output = model(test_x.type('torch.FloatTensor'))
    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)
    """
    sample_submission = pd.DataFrame().reindex_like(trainning_labeling)
    sample_submission.drop(sample_submission.shape[0]-len(predictions))
    # sample_submission.dropna(inplace=True, how='all')
    """
    sample_submission = pd.read_csv(path + '\sample_submission.csv')
    sample_submission['label'] = predictions
    sample_submission.head()
    sample_submission.to_csv(path + '\sample_submission.csv', index=False)
