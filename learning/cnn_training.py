# importing the libraries
import pickle
from datetime import datetime
# for reading and displaying graphs
# from skimage.io import imread
import matplotlib.pyplot as plt
# PyTorch libraries and modules
import torch
# for evaluating the model
from sklearn.metrics import accuracy_score
# for creating validation set
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d

from learning.env import *
from learning.results_manager import *

NUM_OF_CLASSIFICATION_PARAMETERS = 9 # 9 # 3  # 9  # 7
NUM_OF_TIME_SAMPLES = 500 # 100 # 1200 # 300 # 601 # 501  # 301 # 602
DATAFRAME_BEGINNING = 0
DATAFRAME_END = 500
NUM_OF_CONGESTION_CONTROL_LABELING = 3 # 6 # 3
NUM_OF_CONV_FILTERS = 50
# NUM_OF_TRAIN_DATAFRAMES = 3  # 9
# NUM_OF_TEST_DATAFRAMES = 10

def init_weights(model):
    if type(model) == Linear:
        torch.nn.init.xavier_uniform_(model.weight)
        torch.nn.init.zeros_(model.bias)
        # torch.nn.init.xavier_normal(model.weight)
        # torch.nn.init.kaiming_uniform_(model.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        # torch.nn.init.zeros_(model.bias)
        # torch.nn.init.normal_(model.weight, mean=0, std=1)
        #torch.nn.init.uniform(model.weight, 0.0, 1.0)
    # if isinstance(model, Conv2d):
    if type(model) == Conv2d:
        torch.nn.init.xavier_uniform_(model.weight)
        torch.nn.init.zeros_(model.bias)
        #torch.nn.init.kaiming_uniform_(model.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        # conv2d: output size = image_size - filter_size / stride + 1. usually stride = 1, filter [x,x], and padding = (f-1)/2.
        # maxpool2d: output size = image_size - filter_size.
        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, NUM_OF_CONV_FILTERS, kernel_size=(3, NUM_OF_CLASSIFICATION_PARAMETERS), stride=1, padding=(1, 0)), # NUM_OF_CLASSIFICATION_PARAMETERS
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
            Linear(NUM_OF_CONV_FILTERS * 1 * (NUM_OF_TIME_SAMPLES - 2) * 1, NUM_OF_CLASSIFICATION_PARAMETERS)  # 1 instead of NUM_OF_CLASSIFICATION_PARAMETER
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
    """
    if torch.cuda.is_available():
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        x_val = x_val.cuda()
        y_val = y_val.cuda()
    """
    # clearing the Gradients of the model parameters
    optimizer.zero_grad()

    # prediction for training and validation set
    output_train = model(x_train.type('torch.FloatTensor'))
    output_val = model(x_val.type('torch.FloatTensor'))

    # computing the training and validation loss
    """
    softmax = torch.exp(output_train).cpu()
    prob = list(softmax.detach().numpy())
    predictions = np.argmax(prob, axis=1)
    """
    # loss_train = criterion(output_train, y_train.type('torch.LongTensor'))  # Long instead of float (was float and changed to long- now works).
    # loss_val = criterion(output_val, y_val.type('torch.LongTensor'))
    # loss_train = criterion(predictions, y_train.type('torch.LongTensor').view(-1, 1))  # Long instead of float (was float and changed to long- now works).
    # loss_train = criterion(output_train, y_train.type('torch.LongTensor').view(-1, 1))  # Long instead of float (was float and changed to long- now works).
    # loss_train = criterion(output_train, y_train.type('torch.FloatTensor'))  # Long instead of float (was float and changed to long- now works).
    loss_train = criterion(output_train, y_train.type('torch.LongTensor'))  # Long instead of float (was float and changed to long- now works).
    loss_val = criterion(output_val, y_val.type('torch.LongTensor'))
    #train_losses.append(loss_train)
    #val_losses.append(loss_val)

    # saving the training and validation loss
    #pckl_file = open(training_parameters_path + 'model_parameters.pckl', 'wb')
    #pickle.dump([train_losses, val_losses], pckl_file)
    #pckl_file.close()

    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()
    if epoch % 2 == 0:
        # printing the validation loss
        print('Epoch : ', epoch + 1, '\t', 'loss :', loss_val)
    return loss_val

if __name__ == '__main__':
    global model, val_x, val_y, optimizer, criterion, n_epochs, train_losses, val_losses
    # defining the dataframe path
    # normalization_types = ["StatisticalNormalization", "AbsoluteNormalization1", "AbsoluteNormalization2"]
    normalization_types = ["AbsoluteNormalization1"]
    normalization_counter = 0

    # for normalization_type in [StatisticalNormalization(), AbsoluteNormalization1(), AbsoluteNormalization2()]: # 3 different types of normaliztion (pre- processing)
    for normalization_type in [AbsoluteNormalization1()]: # 3 different types of normaliztion (pre- processing)
        res_mgr = ResultsManager(training_files_path, normalization_type, NUM_OF_TIME_SAMPLES, DATAFRAME_BEGINNING, DATAFRAME_END)
        trainning_labeling = res_mgr.get_train_df()
        dataframe_arr = res_mgr.get_normalized_df_list()
        """
        # Added in Saturday:
        num_of_rows = res_mgr.get_num_of_rows()
        NUM_OF_TIME_SAMPLES = num_of_rows
        """

        # Problematic code!!!!
        """
        for csv_file in dataframe_arr: # maybe not necessary
            csv_file = csv_file.drop(csv_file.index[NUM_OF_TIME_SAMPLES:])  # remove samples that were taken after the conventional measuring time.
            csv_file.dropna(inplace=True, how='all')  # remove empty lines after deleting them.
            csv_file = csv_file.fillna((csv_file.shift() + csv_file.shift(-1)) / 2)  # takes care of missing values.
        """
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

        # initializing weights:
        model.apply(init_weights)
        # model.apply(weights_init_uniform)

        # defining the optimizer:
        """
        learning_rate = 1e-4
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        """
        # defining the loss function:
        criterion = CrossEntropyLoss()
        # criterion = L1Loss()
        # criterion = NLLLoss()
        # criterion = MSELoss()
        # criterion = SmoothL1Loss()
        # criterion = KLDivLoss()
        # checking if GPU is available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            """
            model = model.cuda()
            criterion = criterion(reduction="sum").cuda()
            """

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

        for epoch in range(n_epochs):
            #learning_rate = learning_rate_init / (1 + epoch/m)
            #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
            loss_val = train(epoch)
            #scheduler.step(loss_val)
            scheduler.step()
            # if epoch > min_n_epochs and val_losses[epoch] == 0:
            #     break

        # saving the trained model
        torch.save(model, training_parameters_path + normalization_types[normalization_counter] + '_mytraining.pt')
        torch.save(model.state_dict(), training_parameters_path + normalization_types[normalization_counter] + '_mytraining_state_dict.pt')

        # plotting the training and validation loss
        """
        plt.plot(train_losses, label='Training loss')
        plt.plot(val_losses, label='Validation loss')
        plt.legend()
        # plt.show()
        """
        tn = datetime.now()
        time_str = str(tn.month) + "." + str(tn.day) + "." + str(tn.year) + "@" + str(tn.hour) + "-" + str(tn.minute) + "-" + str(tn.second)
        #plt.savefig(training_parameters_path + normalization_types[normalization_counter] + '_graph.jpeg')

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