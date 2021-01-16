import torch
import torch.utils as torch_utils
import numpy
from sklearn.model_selection import train_test_split
# importing project functions
from learning.env import *
from learning.results_manager import *
# consts definitions
NUM_OF_CLASSIFICATION_PARAMETERS = 2 # timestemp & CBIQ
NUM_OF_TIME_SAMPLES = 60000
DEEPCCI_NUM_OF_TIME_SAMPLES = 60
NUM_OF_CONGESTION_CONTROL_LABELING = 3 # Reno, Cubic, & BBR
NUM_OF_CONV_FILTERS = 50
BATCH_SIZE = 32

def create_dataloader(data, labeling):
    dataset = torch_utils.data.TensorDataset(data, labeling)
    dataloader = torch_utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)
    return dataloader

def reshape_deepcci_format(input_data, validation_data, input_labeling, validation_labeling):
    reshape_vector = numpy.ones(DEEPCCI_NUM_OF_TIME_SAMPLES)
    labeling_size = len(input_labeling)
    input_labeling = numpy.kron(input_labeling, reshape_vector)
    input_labeling = input_labeling.reshape(labeling_size, DEEPCCI_NUM_OF_TIME_SAMPLES)
    validation_labeling = numpy.kron(validation_labeling, reshape_vector)
    validation_labeling = validation_labeling.reshape(labeling_size, DEEPCCI_NUM_OF_TIME_SAMPLES)
    # converting training dataframes into torch format
    input_data = torch.from_numpy(input_data)
    input_data = input_data.permute(0, 2, 1)
    # converting the target into torch format
    input_labeling = input_labeling.astype(float)
    input_labeling = torch.from_numpy(input_labeling)
    # converting validation dataframes into torch format
    validation_data = torch.from_numpy(validation_data)
    validation_data = validation_data.permute(0, 2, 1)
    # converting the target into torch format
    validation_labeling = validation_labeling.astype(float)
    validation_labeling = torch.from_numpy(validation_labeling)
    return input_data, validation_data, input_labeling, validation_labeling

def reshape_my_format(input_data, validation_data, input_labeling, validation_labeling):
    # converting training dataframes into torch format
    input_data = input_data.reshape(len(input_data), 1, NUM_OF_TIME_SAMPLES, NUM_OF_CLASSIFICATION_PARAMETERS)
    input_data = torch.from_numpy(input_data)
    # converting the target into torch format
    input_labeling = input_labeling.astype(float)
    input_labeling = torch.from_numpy(input_labeling)
    # converting validation dataframes into torch format
    validation_data = validation_data.reshape(len(validation_data), 1, NUM_OF_TIME_SAMPLES, NUM_OF_CLASSIFICATION_PARAMETERS)
    validation_data = torch.from_numpy(validation_data)
    # converting the target into torch format
    validation_labeling = validation_labeling.astype(float)
    validation_labeling = torch.from_numpy(validation_labeling)
    return input_data, validation_data, input_labeling, validation_labeling

def create_data(training_files_path, normalization_type, is_deepcci):
    result_manager = ResultsManager(training_files_path, normalization_type, NUM_OF_TIME_SAMPLES)
    training_labeling = result_manager.get_train_df()
    input_dataframe = result_manager.get_normalized_df_list()
    # converting the list to numpy array after pre- processing
    input_numpy_dataframe = [dataframe.to_numpy() for dataframe in input_dataframe]
    input_data = np.array(input_numpy_dataframe)
    # defining the target
    input_labeling = np.array(training_labeling['label'].values)
    # creating validation set
    input_data, validation_data, input_labeling, validation_labeling = train_test_split(input_data, input_labeling, test_size=0.1)
    if is_deepcci:
        input_data, validation_data, input_labeling, validation_labeling = reshape_deepcci_format(input_data, validation_data, input_labeling, validation_labeling)
    else:
        input_data, validation_data, input_labeling, validation_labeling = reshape_my_format(input_data, validation_data, input_labeling, validation_labeling)
    # creating dataloaders:
    train_loader = create_dataloader(input_data, input_labeling)
    val_loader = create_dataloader(validation_data, validation_labeling)
    return train_loader, val_loader

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # output = torch.mean(output, -1)
    # target = torch.mean(target, -1)
    res_arr = []
    last_dim_size = output.size(-1)
    result = []
    maxk = max(topk)
    for i in range(last_dim_size):
        curr_output = output[:, :, i:i + 1]
        curr_output = curr_output.squeeze(-1)

        curr_target = target[:, i:i + 1]
        curr_target = curr_target.squeeze(-1)

        with torch.no_grad():

            batch_size = curr_target.size(0)
            _, pred = curr_output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(curr_target.view(1, -1).expand_as(pred))
            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            result.append(res)

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

def accuracy_single_sample(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # output = torch.mean(output, -1)
    # target = torch.mean(target, -1)
    res_arr = []
    result = []
    maxk = max(topk)

    curr_output = output
    curr_target = target

    with torch.no_grad():
        batch_size = curr_target.size(0)
        _, pred = curr_output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(curr_target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        result.append(res)

    return result