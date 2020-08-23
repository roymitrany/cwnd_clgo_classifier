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
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

import glob
import os
import pickle

from learning.environment import *
from learning.results_manager import *


NUM_OF_CLASSIFICATION_PARAMETERS = 9  # 7
NUM_OF_TIME_SAMPLES = 600  # 301 # 602
NUM_OF_CONGESTION_CONTROL_LABELING = 3
NUM_OF_CONV_FILTERS = 10
NUM_OF_TRAIN_DATAFRAMES = 7  # 9
NUM_OF_TEST_DATAFRAMES = 10


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
            Linear(NUM_OF_CONV_FILTERS * (NUM_OF_TIME_SAMPLES - 2) * 1, NUM_OF_CONGESTION_CONTROL_LABELING + 1)
            # an error because the labels must be 0 indexed. So, for example, if you have 20 classes, and the labels are 1th indexed, the 20th label would be 20, so cur_target < n_classes assert would fail. If it’s 0th indexed, the 20th label is 19, so cur_target < n_classes assert passes.
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
    loss_train = criterion(output_train, y_train.type(
        'torch.LongTensor'))  # Long instead of float (was float and changed to long- now works).
    loss_val = criterion(output_val, y_val.type('torch.LongTensor'))
    train_losses.append(loss_train)
    val_losses.append(loss_val)

    # saving the training and validation loss
    pckl_file = open('model_parameters.pckl', 'wb')
    pickle.dump([train_losses, val_losses], pckl_file)
    pckl_file.close()

    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()
    if epoch % 2 == 0:
        # printing the validation loss
        print('Epoch : ', epoch + 1, '\t', 'loss :', loss_val)

if __name__ == '__main__':
    global model, val_x, val_y, optimizer, criterion, n_epochs, train_losses, val_losses
    # defining the dataframe path
    # normalization_types = ["StatisticalNormalization", "AbsoluteNormalization1", "AbsoluteNormalization2"]
    normalization_types = ["AbsoluteNormalization2"]
    normalization_counter = 0

    #for normalization_type in [StatisticalNormalization(), AbsoluteNormalization1(), AbsoluteNormalization2()]: # 3 different types of normaliztion (pre- processing)
    for normalization_type in [AbsoluteNormalization2()]: # 3 different types of normaliztion (pre- processing)
        res_mgr = ResultsManager(training_files_path, normalization_type, NUM_OF_TIME_SAMPLES)
        trainning_labeling = res_mgr.get_train_df()
        dataframe_arr = res_mgr.get_normalized_df_list()
        for csv_file in dataframe_arr: # maybe not necessary
            csv_file = csv_file.drop(csv_file.index[NUM_OF_TIME_SAMPLES:])  # remove samples that were taken after the conventional measuring time.
            csv_file.dropna(inplace=True, how='all')  # remove empty lines after deleting them.
            csv_file = csv_file.fillna((csv_file.shift() + csv_file.shift(-1)) / 2)  # takes care of missing values.

        # converting the list to numpy array after pre- processing
        dataframe_arr = [dataframe.to_numpy() for dataframe in dataframe_arr]
        train_x = np.array(dataframe_arr)
        # defining the target
        train_y = np.array(trainning_labeling['label'].values)
        pckl_file = open(training_parameters_path + normalization_types[normalization_counter] + '.pckl', 'wb')
        pickle.dump([train_x, train_y], pckl_file)
        pckl_file.close()

        # creating validation set
        train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.1)

        # converting training dataframes into torch format
        train_x = train_x.reshape(len(train_x), 1, NUM_OF_TIME_SAMPLES, NUM_OF_CLASSIFICATION_PARAMETERS)
        train_x = torch.from_numpy(train_x)

        # converting the target into torch format
        train_y = train_y.astype(float)
        train_y = torch.from_numpy(train_y)

        # converting validation dataframes into torch format
        val_x = val_x.reshape(len(val_x), 1, NUM_OF_TIME_SAMPLES, NUM_OF_CLASSIFICATION_PARAMETERS)
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

        # saving the trained model
        torch.save(model, training_parameters_path + normalization_types[normalization_counter] + '_mytraining.pt')


        # plotting the training and validation loss
        plt.plot(train_losses, label='Training loss')
        plt.plot(val_losses, label='Validation loss')
        plt.legend()
        # plt.show()
        plt.savefig(training_parameters_path + normalization_types[normalization_counter] + '_graph.jpeg')

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

        normalization_counter +=1