# importing the libraries
import pickle
from datetime import datetime
import torch
import numpy
# for evaluating the model
from sklearn.metrics import accuracy_score
# for creating validation set
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d

from learning.env import *
from learning.results_manager import *
from classification.utils import  *

import torch.nn as nn

NUM_OF_CLASSIFICATION_PARAMETERS = 1
NUM_OF_TIME_SAMPLES = 1024 # 60000
SIZE_OF_TIME_SAMPLE = 3 # 1000
DATAFRAME_BEGINNING = 0
DATAFRAME_END = 1024 # 60000
NUM_OF_CONGESTION_CONTROL_LABELING = 3
NUM_OF_CONV_FILTERS = 50

BATCH_SIZE = 32

def createTrainLoader(train_file):
    normalization_type = AbsoluteNormalization1()
    result_manager = ResultsManager(training_files_path, normalization_type, NUM_OF_TIME_SAMPLES, DATAFRAME_BEGINNING, DATAFRAME_END)
    training_labeling = result_manager.get_train_df()

    input_dataframe = result_manager.get_normalized_df_list()
    # converting the list to numpy array after pre- processing
    input_numpy_dataframe = [dataframe.to_numpy() for dataframe in input_dataframe]
    input_data = np.array(input_numpy_dataframe)
    # defining the target
    input_labeling = np.array(training_labeling['label'].values)

    pckl_file = open(training_parameters_path + "AbsoluteNormalization1" + '.pckl', 'wb')
    pickle.dump([input_data, input_labeling], pckl_file)
    pckl_file.close()
    trainloader = torch.utils.data.DataLoader(input_data, batch_size=BATCH_SIZE)

    return trainloader

class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.channels = 64
        self.num_layers = 100
        self.hidden_size = 64
        # 1
        self.conv1d_layer1_1 = nn.Conv1d(self.channels, self.channels, kernel_size=3, padding=1)
        self.BN_layer1_1 = nn.BatchNorm1d(self.channels)
        self.Relu_layer_1 = nn.ReLU(inplace=False)
        self.conv1d_layer2_1 = nn.Conv1d(self.channels, self.channels, kernel_size=3, padding=1)
        self.BN_layer2_1 = nn.BatchNorm1d(self.channels)
        self.Relu_layer_1 = nn.ReLU(inplace=False)
        self.maxpool_layer_1 = nn.MaxPool1d(kernel_size=4, stride=4, ceil_mode=False)
        # 2
        self.conv1d_layer1_2 = nn.Conv1d(self.channels, self.channels, kernel_size=3, padding=1)
        self.BN_layer1_2 = nn.BatchNorm1d(self.channels)
        self.Relu_layer_2 = nn.ReLU(inplace=False)
        self.conv1d_layer2_2 = nn.Conv1d(self.channels, self.channels, kernel_size=3, padding=1)
        self.BN_layer2_2 = nn.BatchNorm1d(self.channels)
        self.Relu_layer_2 = nn.ReLU(inplace=False)
        self.maxpool_layer_2 = nn.MaxPool1d(kernel_size=4, stride=4, ceil_mode=False)
        # 3
        self.conv1d_layer1_3 = nn.Conv1d(self.channels, self.channels, kernel_size=3, padding=1)
        self.BN_layer1_3 = nn.BatchNorm1d(self.channels)
        self.Relu_layer_3 = nn.ReLU(inplace=False)
        self.conv1d_layer2_3 = nn.Conv1d(self.channels, self.channels, kernel_size=3, padding=1)
        self.BN_layer2_3 = nn.BatchNorm1d(self.channels)
        self.Relu_layer_3 = nn.ReLU(inplace=False)
        self.maxpool_layer_3 = nn.MaxPool1d(kernel_size=4, stride=4, ceil_mode=False)
        # 4
        self.conv1d_layer1_4 = nn.Conv1d(self.channels, self.channels, kernel_size=3, padding=1)
        self.BN_layer1_4 = nn.BatchNorm1d(self.channels)
        self.Relu_layer_4 = nn.ReLU(inplace=False)
        self.conv1d_layer2_4 = nn.Conv1d(self.channels, self.channels, kernel_size=3, padding=1)
        self.BN_layer2_4 = nn.BatchNorm1d(self.channels)
        self.Relu_layer_4 = nn.ReLU(inplace=False)
        self.maxpool_layer_4 = nn.MaxPool1d(kernel_size=4, stride=4, ceil_mode=False)
        # 5
        self.conv1d_layer1_5 = nn.Conv1d(self.channels, self.channels, kernel_size=3, padding=1)
        self.BN_layer1_5 = nn.BatchNorm1d(self.channels)
        self.Relu_layer_5 = nn.ReLU(inplace=False)
        self.conv1d_layer2_5 = nn.Conv1d(self.channels, self.channels, kernel_size=3, padding=1)
        self.BN_layer2_5 = nn.BatchNorm1d(self.channels)
        self.Relu_layer_5 = nn.ReLU(inplace=False)
        self.maxpool_layer_5 = nn.MaxPool1d(kernel_size=4, stride=4, ceil_mode=False)

        # 1
        self.BN_layer_1 = nn.BatchNorm1d(100)
        self.lstm_layer_1 = nn.LSTM(input_size=self.channels, num_layers=1, hidden_size=100)
        # 2
        self.BN_layer_2 = nn.BatchNorm1d(100)
        self.lstm_layer_2 = nn.LSTM(input_size=100, num_layers=1, hidden_size=100)
        # 3
        self.BN_layer_3 = nn.BatchNorm1d(100)
        self.lstm_layer_3 = nn.LSTM(input_size=100, num_layers=1, hidden_size=100)

        self.init_fg_bias()

        self.conv1 = nn.Conv1d(1, self.channels, kernel_size=1)

        self.first_BN_layer = (nn.BatchNorm1d(self.channels))
        self.last_conv1d_layer = nn.Conv1d(100, 5, kernel_size=1, padding=1)
        self.last_BN_layer = nn.BatchNorm1d(5)

    def init_fg_bias(self):
        for names in self.lstm_layer_1._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm_layer_1, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)
        for names in self.lstm_layer_2._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm_layer_2, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)
        for names in self.lstm_layer_3._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm_layer_3, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

    # Defining the forward pass
    def forward(self, x):
        #x = torch.zeros([180, 1, 60000], dtype=torch.int32)
        #x=x.type('torch.FloatTensor')

        x = self.conv1(x)
        # 1
        residual = x
        x = self.conv1d_layer1_1(x)
        x = self.BN_layer1_1(x)
        x = self.Relu_layer_1(x)
        x = self.conv1d_layer2_1(x)
        x = self.BN_layer2_1(x)
        x = x + residual
        x = self.Relu_layer_1(x)
        x = self.maxpool_layer_1(x)
        # 2
        residual = x
        x = self.conv1d_layer1_2(x)
        x = self.BN_layer1_2(x)
        x = self.Relu_layer_2(x)
        x = self.conv1d_layer2_2(x)
        x = self.BN_layer2_2(x)
        x = x + residual
        x = self.Relu_layer_2(x)
        x = self.maxpool_layer_2(x)
        # 3
        residual = x
        x = self.conv1d_layer1_3(x)
        x = self.BN_layer1_3(x)
        x = self.Relu_layer_3(x)
        x = self.conv1d_layer2_3(x)
        x = self.BN_layer2_3(x)
        x = x + residual
        x = self.Relu_layer_3(x)
        x = self.maxpool_layer_3(x)
        # 4
        residual = x
        x = self.conv1d_layer1_4(x)
        x = self.BN_layer1_4(x)
        x = self.Relu_layer_4(x)
        x = self.conv1d_layer2_4(x)
        x = self.BN_layer2_4(x)
        x = x + residual
        x = self.Relu_layer_4(x)
        x = self.maxpool_layer_4(x)
        # 5
        residual = x
        x = self.conv1d_layer1_5(x)
        x = self.BN_layer1_5(x)
        x = self.Relu_layer_5(x)
        x = self.conv1d_layer2_5(x)
        x = self.BN_layer2_5(x)
        x = x + residual
        x = self.Relu_layer_5(x)
        x = self.maxpool_layer_5(x)

        x = self.first_BN_layer(x)
        # 1
        x = x.transpose(1, 2)
        x = x.transpose(0, 1)
        x, _ = self.lstm_layer_1(x)
        x = x.transpose(0, 1)
        x = x.transpose(1, 2)
        x = self.BN_layer_1(x)
        # 2
        x = x.transpose(1, 2)
        x = x.transpose(0, 1)
        x, _ = self.lstm_layer_2(x)
        x = x.transpose(0, 1)
        x = x.transpose(1, 2)
        x = self.BN_layer_2(x)
        # 3
        x = x.transpose(1, 2)
        x = x.transpose(0, 1)
        x, _ = self.lstm_layer_3(x)
        x = x.transpose(0, 1)
        x = x.transpose(1, 2)
        x = self.BN_layer_3(x)

        x = self.last_conv1d_layer(x)
        x = self.last_BN_layer(x)

        x_variant = x[:, 0:3, :]
        x_pacing = x[:, 3:5, :]

        #pacing_output = self.pacing_softmax_layer(x_pacing)
        #variant_output = self.variant_softmax_layer(x_variant)

        pacing_output = x_pacing
        variant_output = x_variant

        return variant_output, pacing_output
        return x

def train(epoch):
    #training_accuracy = AverageMeter('training', 'acc')

    model.train()
    x_train, y_train = Variable(input_data), Variable(input_labeling)
    # getting the validation set
    x_val, y_val = Variable(validation_data), Variable(validation_labeling)

    # prediction for training and validation set
    output_train, hidden = model(x_train.type('torch.FloatTensor'))
    #output_val = model(x_val.type('torch.FloatTensor'))
    output_val, hidden = model(x_val.type('torch.FloatTensor'))

    #expected_output = y_train.type('torch.LongTensor')
    #expected_output = torch.unsqueeze(expected_output, 1)
    loss_train = criterion(output_train, y_train.type('torch.LongTensor'))  # Long instead of float (was float and changed to long- now works).
    loss_val = criterion(output_val, y_val.type('torch.LongTensor'))


    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()
    if epoch % 2 == 0:
        # printing the validation loss
        print('Epoch : ', epoch + 1, '\t', 'loss :', loss_val)

    training_accuracy, hidden = accuracy(output_train, input_labeling, topk=(1, 3))

    return loss_val, training_accuracy

if __name__ == '__main__':
    global model, validation_data, validation_labeling, optimizer, criterion, n_epochs, train_losses, val_losses
    # defining the dataframe path
    normalization_type = AbsoluteNormalization1()
    result_manager = ResultsManager(training_files_path, normalization_type, NUM_OF_TIME_SAMPLES, DATAFRAME_BEGINNING,
                                    DATAFRAME_END)
    training_labeling = result_manager.get_train_df()

    input_dataframe = result_manager.get_normalized_df_list()
    # converting the list to numpy array after pre- processing
    input_numpy_dataframe = [dataframe.to_numpy() for dataframe in input_dataframe]
    input_data = np.array(input_numpy_dataframe)
    # defining the target
    input_labeling = np.array(training_labeling['label'].values)

    pckl_file = open(training_parameters_path + "AbsoluteNormalization1" + '.pckl', 'wb')
    pickle.dump([input_data, input_labeling], pckl_file)
    pckl_file.close()

    # creating validation set
    input_data, validation_data, input_labeling, validation_labeling = train_test_split(input_data, input_labeling, test_size=0.1)

    reshape_vector = numpy.ones(3) # 60
    labeling_size = len(input_labeling)
    input_labeling = numpy.kron(input_labeling, reshape_vector)
    input_labeling = input_labeling.reshape(labeling_size, SIZE_OF_TIME_SAMPLE)

    labeling_size = len(validation_labeling)
    validation_labeling = numpy.kron(validation_labeling, reshape_vector)
    validation_labeling = validation_labeling.reshape(labeling_size, SIZE_OF_TIME_SAMPLE)

    # converting training dataframes into torch format
    #train_x = train_x.reshape(len(train_x), 1, NUM_OF_TIME_SAMPLES, NUM_OF_CLASSIFICATION_PARAMETERS)
    input_data = torch.from_numpy(input_data)
    input_data = input_data.permute(0, 2, 1)
    # converting the target into torch format
    input_labeling = input_labeling.astype(float)
    input_labeling = torch.from_numpy(input_labeling)
    # converting validation dataframes into torch format
    #val_x = val_x.reshape(len(val_x), 1, NUM_OF_TIME_SAMPLES, NUM_OF_CLASSIFICATION_PARAMETERS)
    validation_data = torch.from_numpy(validation_data)
    validation_data = validation_data.permute(0, 2, 1)
    # converting the target into torch format
    validation_labeling = validation_labeling.astype(float)
    validation_labeling = torch.from_numpy(validation_labeling)

    # defining the model
    model = Net()

    # initializing weights:
    #model.apply(init_weights)

    # defining the loss function:
    criterion = CrossEntropyLoss()
    # defining the number of epochs
    n_epochs = 100 # 75 # 50 # 70 # 50 # 25
    min_n_epochs = 10 # 25
    # empty list to store training losses
    train_losses = []
    # empty list to store validation losses
    val_losses = []
    # training the model:
    m = 25
    learning_rate_init = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5, threshold=0.01, threshold_mode='abs')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.9)

    training_accuracy = []
    for epoch in range(n_epochs):
        #learning_rate = learning_rate_init / (1 + epoch/m)
        #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_val, accuracy_training = train(epoch)
        training_accuracy.append(accuracy_training)
        #scheduler.step(loss_val)
        scheduler.step()

    # saving the trained model
    torch.save(model, training_parameters_path + "AbsoluteNormalization1" + '_mytraining.pt')
    torch.save(model.state_dict(), training_parameters_path + "AbsoluteNormalization1" + '_mytraining_state_dict.pt')

    tn = datetime.now()
    time_str = str(tn.month) + "." + str(tn.day) + "." + str(tn.year) + "@" + str(tn.hour) + "-" + str(tn.minute) + "-" + str(tn.second)

    # prediction for training set
    with torch.no_grad():
        output = model(input_data.type('torch.FloatTensor'))

    """
    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)

    # accuracy on training set
    print(accuracy_score(input_labeling, predictions))

    # prediction for validation set
    with torch.no_grad():
        output = model(validation_data.type('torch.FloatTensor'))

    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)

    # accuracy on validation set
    print(accuracy_score(validation_labeling, predictions))
    """
    # training_accuracy, hidden = accuracy(output, input_labeling, topk=(1, 3))



