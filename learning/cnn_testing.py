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
import pickle

from learning.env import *
from learning.cnn_training import Net
from learning.results_manager import *
import time
NUM_OF_CLASSIFICATION_PARAMETERS = 9  # 7
NUM_OF_TIME_SAMPLES = 601  # 301 # 602
NUM_OF_CONGESTION_CONTROL_LABELING = 3
NUM_OF_CONV_FILTERS = 10
NUM_OF_TRAIN_DATAFRAMES = 3  # 9
NUM_OF_TEST_DATAFRAMES = 3

if __name__ == '__main__':
    global model, val_x, val_y, optimizer, criterion, n_epochs, train_losses, val_losses

    normalization_types = ["StatisticalNormalization", "AbsoluteNormalization1", "AbsoluteNormalization2"]
    normalization_counter = 0
    classification_difference = []
    for normalization_type in [StatisticalNormalization(), AbsoluteNormalization1(), AbsoluteNormalization2()]: # 3 different types of normaliztion (pre- processing)
        model = torch.load(training_parameters_path + normalization_types[normalization_counter] + '_mytraining.pt')  # loading the model and its parameters.
        res_mgr = ResultsManager(testing_files_path, normalization_type, NUM_OF_TIME_SAMPLES)
        trainning_labeling = res_mgr.get_train_df()
        dataframe_arr = res_mgr.get_normalized_df_list()
        for csv_file in dataframe_arr: # maybe not necessary
            csv_file = csv_file.drop(csv_file.index[NUM_OF_TIME_SAMPLES:])  # remove samples that were taken after the conventional measuring time.
            csv_file.dropna(inplace=True, how='all')  # remove empty lines after deleting them.
            csv_file = csv_file.fillna((csv_file.shift() + csv_file.shift(-1)) / 2)  # takes care of missing values.

        test_x = [dataframe.to_numpy() for dataframe in dataframe_arr]
        test_x = np.array(test_x)
        test_x = test_x.reshape(len(test_x), 1, NUM_OF_TIME_SAMPLES, NUM_OF_CLASSIFICATION_PARAMETERS)
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
        testing_labeling = trainning_labeling.copy()
        testing_labeling['label'] = predictions
        testing_labeling.head()
        trainning_labeling.to_csv(testing_results_path + '\\' + normalization_types[normalization_counter] + time.strftime("_%d_%m_%Y") + '_training_labeling.csv', index=False)
        testing_labeling.to_csv(testing_results_path + '\\' + normalization_types[normalization_counter] + time.strftime("_%d_%m_%Y") + '_testing_labeling.csv', index=False)

        # comparing "train" and "sample_submission":
        classification_difference.append((testing_labeling[testing_labeling != trainning_labeling].count()[1])/len(testing_labeling))
        # print(classification_difference.count()[1] / len(testing_labeling))
        normalization_counter+=1

    print(classification_difference)