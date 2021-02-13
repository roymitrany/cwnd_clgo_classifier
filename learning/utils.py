import os
import torch
import torch.utils as torch_utils
import numpy
from sklearn.model_selection import train_test_split
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d
from torch.autograd import Variable
import matplotlib.pyplot as plt
# import project functions
from learning.env import *
from learning.results_manager import *
# consts definitions
NUM_OF_CLASSIFICATION_PARAMETERS = 2 # timestemp & CBIQ
NUM_OF_TIME_SAMPLES = 60000
DEEPCCI_NUM_OF_TIME_SAMPLES = 60
NUM_OF_CONGESTION_CONTROL_LABELING = 3 # Reno, Cubic, & BBR
NUM_OF_CONV_FILTERS = 50
NUM_OF_HIDDEN_LAYERS = 100
BATCH_SIZE = 32
TRAINING_VALIDATION_RATIO = 0.3
# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def reshape_deepcci_data(input_data, validation_data, input_labeling, validation_labeling):
    reshape_vector = numpy.ones(DEEPCCI_NUM_OF_TIME_SAMPLES)
    labeling_size = len(input_labeling)
    input_labeling = numpy.kron(input_labeling, reshape_vector)
    input_labeling = input_labeling.reshape(labeling_size, DEEPCCI_NUM_OF_TIME_SAMPLES)
    validation_size = len(validation_data)
    validation_labeling = numpy.kron(validation_labeling, reshape_vector)
    validation_labeling = validation_labeling.reshape(validation_size, DEEPCCI_NUM_OF_TIME_SAMPLES)
    # converting training dataframes into torch format
    input_data = torch.from_numpy(input_data)
    #input_data = input_data.permute(0, 2, 1)
    input_data = input_data.reshape(len(input_data), 1, NUM_OF_TIME_SAMPLES)
    # converting the target into torch format
    input_labeling = torch.from_numpy(input_labeling)
    # converting validation dataframes into torch format
    validation_data = torch.from_numpy(validation_data)
    #validation_data = validation_data.permute(0, 2, 1)
    validation_data = validation_data.reshape(len(validation_data), 1, NUM_OF_TIME_SAMPLES)
    # converting the target into torch format
    validation_labeling = torch.from_numpy(validation_labeling)
    return input_data, validation_data, input_labeling, validation_labeling

def reshape_my_data(input_data, validation_data, input_labeling, validation_labeling):
    # converting training dataframes into torch format
    input_data = input_data.reshape(len(input_data), 1, NUM_OF_TIME_SAMPLES, NUM_OF_CLASSIFICATION_PARAMETERS)
    input_data = torch.from_numpy(input_data)  # convert to tensor
    # converting the target into torch format
    input_labeling = torch.from_numpy(input_labeling)
    # converting validation dataframes into torch format
    validation_data = validation_data.reshape(len(validation_data), 1, NUM_OF_TIME_SAMPLES, NUM_OF_CLASSIFICATION_PARAMETERS)
    validation_data = torch.from_numpy(validation_data)
    # converting the target into torch format
    validation_labeling = torch.from_numpy(validation_labeling)
    return input_data, validation_data, input_labeling, validation_labeling

def create_dataloader(data, labeling, is_batch):
    dataset = torch_utils.data.TensorDataset(data, labeling)
    if is_batch:
        dataloader = torch_utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)
    else:
        dataloader = torch_utils.data.DataLoader(dataset, batch_size=len(data))
    return dataloader

def create_data(training_files_path, normalization_type, unused_parameters, is_deepcci, is_batch):
    result_manager = ResultsManager(training_files_path, normalization_type, NUM_OF_TIME_SAMPLES, unused_parameters)
    training_labeling = result_manager.get_train_df()
    input_dataframe = result_manager.get_normalized_df_list()
    # converting the list to numpy array after pre- processing
    input_numpy_dataframe = [dataframe.to_numpy() for dataframe in input_dataframe]
    input_data = np.array(input_numpy_dataframe)
    # defining the target
    input_labeling = np.array(training_labeling['label'].values)
    # creating validation set
    input_data, validation_data, input_labeling, validation_labeling = train_test_split(input_data, input_labeling, test_size=TRAINING_VALIDATION_RATIO)
    if is_deepcci:
        input_data, validation_data, input_labeling, validation_labeling = reshape_deepcci_data(input_data, validation_data, input_labeling, validation_labeling)
    else:
        input_data, validation_data, input_labeling, validation_labeling = reshape_my_data(input_data, validation_data, input_labeling, validation_labeling)
    # creating dataloaders:
    train_loader = create_dataloader(input_data, input_labeling, is_batch)
    val_loader = create_dataloader(validation_data, validation_labeling, is_batch)
    return train_loader, val_loader

def accuracy_aux(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)

    curr_output = output
    curr_target = target

    with torch.no_grad():
        batch_size = curr_target.size(0)
        _, pred = curr_output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(curr_target.view(1, -1).expand_as(pred))
        if len(topk) == 1:
            correct_k = correct[:1].view(-1).float().sum(0, keepdim=True)
            return correct_k.mul_(100.0 / batch_size)
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res

def accuracy(output, target, topk, is_deepcci):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    last_dim_size = output.size(-1)
    result = []
    maxk = max(topk)
    if not is_deepcci:
        curr_output = output
        curr_target = target
        return accuracy_aux(curr_output, curr_target, topk)
    else:
        for i in range(last_dim_size):
            curr_output = output[:, :, i:i + 1]
            curr_output = curr_output.squeeze(-1)
            curr_target = target[:, i:i + 1]
            curr_target = curr_target.squeeze(-1)
            res = accuracy_aux(curr_output, curr_target, topk)
            result.append(res)
    if len(topk) == 1:
        return torch.mean(torch.FloatTensor(result))
    result_summary = []
    for i in range(2):
        result_summary.append([])
    for i in range(last_dim_size):
        for j, elem in enumerate(result[i]):
            result_summary[j].append(elem)
    for i in range(2):
        result_summary[i] = torch.FloatTensor(result_summary[i])
        result_summary[i] = [torch.mean(result_summary[i], -1)]
    return result_summary


class Graph_Creator:
    def __init__(self, loss, accuracy, num_of_epochs, is_batch, plot_file_name="Graph.png", plot_fig_name="Statistics"):
        #loss = np.array(loss, dtype=np.float32)
        #accuracy = np.array(accuracy, dtype=np.float32)
        # clear plt before starting new statistics, otherwise it add up to the previous one
        self.is_batch = is_batch
        self.loss = loss
        self.accuracy = accuracy
        plt.cla()  # clear the current axes
        plt.clf()  # clear the current figure
        fig1, (ax1, ax2) = plt.subplots(2, constrained_layout=True)
        fig1.suptitle(plot_fig_name)
        ax1.set(xlabel='epoch', ylabel='loss')
        ax1.set_xlim([0, num_of_epochs])
        ax1.set_ylim([0, numpy.amax(loss)])
        ax2.set(xlabel='epoch', ylabel='accuracy')
        ax2.set_xlim([0, num_of_epochs])
        ax2.set_ylim([0, numpy.amax(accuracy)])
        self.fig1 = fig1
        self.loss_ax = ax1
        self.accuracy_ax = ax2
        if self.is_batch:
            self.average_loss = numpy.mean(loss, axis=1)
            self.average_accuracy = numpy.mean(accuracy, axis=1)
            fig2, (ax3, ax4) = plt.subplots(2, constrained_layout=True)
            fig2.suptitle(plot_fig_name)
            ax3.set(xlabel='epoch', ylabel='loss')
            ax3.set_xlim([0, num_of_epochs])
            ax3.set_ylim([0, numpy.amax(loss)])
            ax4.set(xlabel='epoch', ylabel='accuracy')
            ax4.set_xlim([0, num_of_epochs])
            ax4.set_ylim([0, numpy.amax(accuracy)])
            self.fig2 = fig2
            self.batches_loss_ax = ax3
            self.batches_accuracy_ax = ax4
        self.plot_file_name = plot_file_name

    def create_loss_plot(self, loss, ax):
        loss_df = pd.DataFrame(loss)#.transpose()
        loss_df.plot(kind='line', ax=ax, title='loss')
        self.write_to_file("loss", loss)

    def create_accuracy_plot(self, accuracy, ax):
        # accuracy = accuracy.squeeze()
        accuracy_df = pd.DataFrame(accuracy)#.transpose()
        accuracy_df.plot(kind='line', ax=ax, title='accuracy')
        self.write_to_file("accuracy", accuracy)

    def create_graphs(self):
        if self.is_batch:
            self.create_loss_plot(self.average_loss, self.loss_ax)
            self.create_accuracy_plot(self.average_accuracy, self.accuracy_ax)
            self.loss_ax.grid()
            self.accuracy_ax.grid()
            self.save_and_show(self.fig1)
            self.plot_file_name = self.plot_file_name.replace('.png', ('_batches_figures.png'))
            self.create_loss_plot(self.loss, self.batches_loss_ax)
            self.create_accuracy_plot(self.accuracy, self.batches_accuracy_ax)
            self.batches_loss_ax.grid()
            self.batches_accuracy_ax.grid()
            self.save_and_show(self.fig2)
        else:
            self.create_loss_plot(self.loss, self.loss_ax)
            self.create_accuracy_plot(self.accuracy, self.accuracy_ax)
            self.loss_ax.grid()
            self.accuracy_ax.grid()
            self.save_and_show(self.fig1)

    def write_to_file(self, type, list):
        with open(self.plot_file_name.replace('.png', ('_' + type)), 'w') as f:
            for item in list:
                f.write("%s\n" % item)

    def save_and_show(self, fig):
        fig.savefig(self.plot_file_name, dpi=600)
        # plt.show()
        plt.close(fig)


    """
    def create_loss_plot(self, loss):
        loss = [l[-1] for l in loss]
        loss_df = pd.DataFrame(loss)#.transpose()
        loss_df.plot(kind='line', ax=self.loss_ax, title='loss')
        self.write_to_file("loss", loss)
        
    def create_accuracy_plot(self, accuracy):
        # accuracy = accuracy.squeeze()
        accuracy = [a[-1] for a in accuracy]
        accuracy_df = pd.DataFrame(accuracy)#.transpose()
        accuracy_df.plot(kind='line', ax=self.accuracy_ax, title='accuracy')
        self.write_to_file("accuracy", accuracy)
    """