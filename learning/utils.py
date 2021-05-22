import os
import torch
import torch.utils as torch_utils
import numpy
import re
import statistics
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d
from torch.autograd import Variable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import project functions
from learning.env import *
from learning.results_manager import *
"""
# consts definitions
NUM_OF_CLASSIFICATION_PARAMETERS = 2 # timestemp & CBIQ
NUM_OF_TIME_SAMPLES = 60000
DEEPCCI_NUM_OF_TIME_SAMPLES = 60
NUM_OF_CONGESTION_CONTROL_LABELING = 3 # Reno, Cubic, & BBR
NUM_OF_CONV_FILTERS = 50
NUM_OF_HIDDEN_LAYERS = 100
BATCH_SIZE = 32
TRAINING_VALIDATION_RATIO = 0.3
"""
from learning.env import *
from numpy import loadtxt


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
    input_data = input_data.reshape(len(input_data), 1, CHUNK_SIZE)
    # converting the target into torch format
    input_labeling = torch.from_numpy(input_labeling)
    # converting validation dataframes into torch format
    validation_data = torch.from_numpy(validation_data)
    #validation_data = validation_data.permute(0, 2, 1)
    validation_data = validation_data.reshape(len(validation_data), 1, CHUNK_SIZE)
    # converting the target into torch format
    validation_labeling = torch.from_numpy(validation_labeling)
    return input_data, validation_data, input_labeling, validation_labeling

def reshape_my_data(input_data, validation_data, input_labeling, validation_labeling):
    # converting training dataframes into torch format
    input_data = input_data.reshape(len(input_data), 1, CHUNK_SIZE, NUM_OF_CLASSIFICATION_PARAMETERS)
    input_data = torch.from_numpy(input_data)  # convert to tensor
    # converting the target into torch format
    input_labeling = torch.from_numpy(input_labeling)
    # converting validation dataframes into torch format
    validation_data = validation_data.reshape(len(validation_data), 1, CHUNK_SIZE, NUM_OF_CLASSIFICATION_PARAMETERS)
    validation_data = torch.from_numpy(validation_data)
    # converting the target into torch format
    validation_labeling = torch.from_numpy(validation_labeling)
    return input_data, validation_data, input_labeling, validation_labeling

def create_dataloader(data, labeling, is_batch):
    dataset = torch_utils.data.TensorDataset(data, labeling)
    if is_batch:
        dataloader = torch_utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=IS_SHUFFLE)#batch_size=int(len(data) / NUM_OF_BATCHES))
    else:
        dataloader = torch_utils.data.DataLoader(dataset, batch_size=len(data))
    return dataloader

def create_data(training_files_path, normalization_type, unused_parameters, is_deepcci, is_batch, diverse_training_folder):
    result_manager = ResultsManager(training_files_path, normalization_type, NUM_OF_TIME_SAMPLES, unused_parameters, chunk_size=CHUNK_SIZE, start_after=START_AFTER, end_before=END_BEFORE, is_diverse = IS_DIVERSE, diverse_training_folder=diverse_training_folder)
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

def accuracy_aux(output, target, topk=(1,), is_f1=True):
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
            if is_f1:
                return f1_score(target.cpu(), np.argmax(output.cpu(), axis=1), average='macro')
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

def accuracy_per_type(output, target, topk=(1,)):
    maxk = max(topk)
    curr_output = output
    curr_target = target
    num_of_classes = len(output[0])
    with torch.no_grad():
        batch_size = curr_target.size(0)
        _, pred = curr_output.topk(maxk, 1, True, True)
        pred = pred.t()
        acc = [0 for c in range(num_of_classes)]
        for c in range(num_of_classes):
            num_of_traget = curr_target == c
            correct = pred.eq(curr_target.view(1, -1).expand_as(pred)) * num_of_traget
            acc[c] = correct[:1].view(-1).float().sum(0, keepdim=True).mul_(100.0 / (num_of_traget.sum())).item()
            #acc[c] = ((pred == curr_target) * (curr_target == c)).float() / (max(curr_target == c).sum(), 1)
    return acc


def get_accuracy_vs_session_duration(results_path, txt_filename, plot_name, single_session_duration):
    my_net_accuracy_list = []
    my_net_all_parameters_accuracy_list = []
    deepcci_net_accuracy_list = []
    for dir_name in os.listdir(results_path):
        session_duration = re.findall(r'\d+', dir_name)
        res_dir = os.path.join(results_path, dir_name)
        if not os.path.isdir(res_dir):
            continue
        if single_session_duration is not None:
            if len(re.findall(str(single_session_duration)+'_', res_dir)) == 0:
                continue
        res_file = os.path.join(res_dir, txt_filename)
        with open(res_file) as f:
            accuracy = f.readlines()
            accuracy = [x.strip() for x in accuracy]
            accuracy = [float(i) for i in accuracy]
            if "my_net" in res_file:
                if "all_parameters" in res_file:
                    my_net_all_parameters_accuracy_list.append((statistics.mean(accuracy[80:]), int(session_duration[0]) / 1000))
                else:
                    my_net_accuracy_list.append((statistics.mean(accuracy[80:]), int(session_duration[0]) / 1000))
            else:
                deepcci_net_accuracy_list.append((statistics.mean(accuracy[80:]), int(session_duration[0]) / 1000))
    return my_net_accuracy_list, my_net_all_parameters_accuracy_list, deepcci_net_accuracy_list


def create_acuuracy_vs_session_duration_graph(results_path, txt_filename, plot_name):
    my_net_accuracy_list, my_net_all_parameters_accuracy_list, deepcci_net_accuracy_list = get_accuracy_vs_session_duration(results_path, txt_filename, plot_name, None)
    plt.cla()  # clear the current axes
    plt.clf()  # clear the current figure
    my_net_accuracy_list = sorted(my_net_accuracy_list, key=lambda tup: tup[1])
    session_duration = [x[1] for x in my_net_accuracy_list]
    accuracy = [x[0] for x in my_net_accuracy_list]
    p1, = plt.plot(session_duration, accuracy)
    plt.title(plot_name)
    if my_net_all_parameters_accuracy_list:
        my_net_all_parameters_accuracy_list = sorted(my_net_all_parameters_accuracy_list, key=lambda tup: tup[1])
        session_duration = [x[1] for x in my_net_all_parameters_accuracy_list]
        accuracy = [x[0] for x in my_net_all_parameters_accuracy_list]
        p2, = plt.plot(session_duration, accuracy)
    if deepcci_net_accuracy_list:
        deepcci_net_accuracy_list = sorted(deepcci_net_accuracy_list, key=lambda tup: tup[1])
        session_duration = [x[1] for x in deepcci_net_accuracy_list]
        accuracy = [x[0] for x in deepcci_net_accuracy_list]
        p3, = plt.plot(session_duration, accuracy)
    if p2 is not None and p3 is not None:
        plt.legend((p1, p2, p3), ('my_net','my_net_all_parameters', 'deepcci_net'))
    else:
        if p3 is not None:
            plt.legend((p1, p3), ('my_net', 'deepcci_net'))
    axes = plt.gca()
    axes.set(xlabel='session duration[seconds]', ylabel='accuracy')
    #axes.set_xlim([0, len(my_net_accuracy_list)])
    #plt.xticks(session_duration)
    #axes.set_ylim([0, numpy.amax(my_net_accuracy_list)])
    axes.grid()
    plt.savefig(os.path.join(results_path, plot_name), dpi=600)
    """
    from learning.utils import *
    #result_path="/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/unfixed_session_duration/30_background_tcp_flows"
    result_path="/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/Thesis/Time_Interval_15_flows"
    create_acuuracy_vs_session_duration_graph(result_path,"validation_accuracy","accuracy_vs_session_duration", [1, 3, 6, 10, 30, 60])
    """

def create_acuuracy_vs_number_of_flows_graph(results_path, txt_filename, plot_name, session_duration):
    my_net_accuracy_list = []
    my_net_all_parameters_accuracy_list = []
    deepcci_net_accuracy_list = []
    for dir_name in os.listdir(results_path):
        number_of_flows = re.findall(r'\d+', dir_name)
        res_dir = os.path.join(results_path, dir_name)
        if not os.path.isdir(res_dir):
            continue
        my_net_accuracy, my_net_all_parameters_accuracy, deepcci_net_accuracy = get_accuracy_vs_session_duration(res_dir, txt_filename, plot_name, session_duration)
        if my_net_accuracy:
            my_net_accuracy_list_accuracy = [x[0] for x in my_net_accuracy]
            my_net_accuracy_list.append((int(number_of_flows[0]), my_net_accuracy_list_accuracy[0]))
        if my_net_all_parameters_accuracy:
            my_net_all_parameters_accuracy_list_accuracy = [x[0] for x in my_net_all_parameters_accuracy]
            my_net_all_parameters_accuracy_list.append((int(number_of_flows[0]), my_net_all_parameters_accuracy_list_accuracy[0]))
        if deepcci_net_accuracy:
            deepcci_net_accuracy_list_accuracy = [x[0] for x in deepcci_net_accuracy]
            deepcci_net_accuracy_list.append((int(number_of_flows[0]), deepcci_net_accuracy_list_accuracy[0]))
    plt.cla()  # clear the current axes
    plt.clf()  # clear the current figure
    plt.title(plot_name)
    my_net_accuracy_list = sorted(my_net_accuracy_list, key=lambda tup: tup[0])
    number_of_flows = [x[0] for x in my_net_accuracy_list]
    accuracy = [x[1] for x in my_net_accuracy_list]
    p1, = plt.plot(number_of_flows, accuracy)
    p2=p3=None
    if my_net_all_parameters_accuracy_list:
        my_net_all_parameters_accuracy_list = sorted(my_net_all_parameters_accuracy_list, key=lambda tup: tup[0])
        number_of_flows = [x[0] for x in my_net_all_parameters_accuracy_list]
        accuracy = [x[1] for x in my_net_all_parameters_accuracy_list]
        p2, = plt.plot(number_of_flows, accuracy)
    if deepcci_net_accuracy_list:
        deepcci_net_accuracy_list = sorted(deepcci_net_accuracy_list, key=lambda tup: tup[0])
        number_of_flows = [x[0] for x in deepcci_net_accuracy_list]
        accuracy = [x[1] for x in deepcci_net_accuracy_list]
        p3, = plt.plot(number_of_flows, accuracy)
    if p2 is not None and p3 is not None:
        plt.legend((p1, p2, p3), ('my_net','my_net_all_parameters', 'deepcci_net'))
    else:
        if p3 is not None:
            plt.legend((p1, p3), ('my_net', 'deepcci_net'))
    axes = plt.gca()
    axes.set(xlabel='number of flows', ylabel='accuracy')
    #axes.set_xlim([0, len(my_net_accuracy_list)])
    #plt.xticks(number_of_flows)
    #axes.set_ylim([0, numpy.amax(my_net_accuracy_list)])
    axes.grid()
    plt.savefig(os.path.join(results_path, plot_name), dpi=600)
    """
    from learning.utils import *
    result_path="/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/unfixed_session_duration/bbr_cubic_reno_background_tcp_flows/"
    create_acuuracy_vs_number_of_flows_graph(result_path,"validation_accuracy","accuracy_vs_number_of_flows_60seconds_session_duration", [0, 15, 30, 75], 60000)
    """


def create_3d_graph_from_file(results_path, txt_filename, plot_name, x_values, y_values):
    my_net_accuracy_list = []
    deepcci_net_accuracy_list = []
    X = []
    Y = []
    for dir_name in os.listdir(results_path):
        res_dir = os.path.join(results_path, dir_name)
        if not os.path.isdir(res_dir):
            continue
        my_net_accuracy_list_temp, deepcci_net_accuracy_list_temp = get_accuracy_vs_session_duration(res_dir, txt_filename, plot_name, x_values)
        my_net_accuracy_list.append(my_net_accuracy_list_temp)
        deepcci_net_accuracy_list.append(deepcci_net_accuracy_list_temp[::-1])
        X.append(x_values)
        Y.append(y_values)
    plt.cla()  # clear the current axes
    plt.clf()  # clear the current figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #X, Y = np.meshgrid(x_values, y_values)
    Z1 = np.array(my_net_accuracy_list)
    Z2 = np.array(deepcci_net_accuracy_list)
    ax.plot_surface(X, Y, Z1)
    ax.plot_surface(X, Y, Z2)
    ax.set_xlabel('session duration[seconds]')
    ax.set_ylabel('number of background flows')
    ax.set_zlabel('accuracy')
    plt.title(plot_name)
    plt.legend((Z1, Z2), ('my_net', 'deepcci_net'))
    plt.savefig(os.path.join(results_path, plot_name), dpi=600)
    """
    from learning.utils import *
    result_path="/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/unfixed_session_duration/"
    create_3d_graph_from_file(result_path,"validation_accuracy","accuracy_vs_session_duration", [1, 3, 6, 10, 30, 60], [0, 15, 30, 75])
    """

def create_accuracy_vs_epoch_graph(results_path, txt_filename, plot_name, short_epoch):
    my_net_accuracy = []
    my_net_all_parameters_accuracy = []
    deepcci_net_accuracy = []
    for dir_name in os.listdir(results_path):
        session_duration = re.findall(r'\d+', dir_name)
        res_dir = os.path.join(results_path, dir_name)
        if not os.path.isdir(res_dir):
            continue
        res_file = os.path.join(res_dir, txt_filename)
        with open(res_file) as f:
            accuracy = f.readlines()
            accuracy = [x.strip() for x in accuracy]
            accuracy = [float(i) for i in accuracy]
            if "my_net_chunk" in res_file:
                if "all_parameters" in res_file:
                    my_net_all_parameters_accuracy = accuracy
                else:
                    my_net_accuracy = accuracy
            else:
                deepcci_net_accuracy = accuracy
    plt.cla()  # clear the current axes
    plt.clf()  # clear the current figure
    if short_epoch:
        p1, = plt.plot(my_net_accuracy[0:55])
        p2, = plt.plot(my_net_all_parameters_accuracy[0:55])
        p3, = plt.plot(deepcci_net_accuracy[0:55])
    else:
        p1, = plt.plot(my_net_accuracy)
        p2, = plt.plot(my_net_all_parameters_accuracy)
        p3, = plt.plot(deepcci_net_accuracy)
    plt.legend((p1, p2, p3), ('my_net','my_net_all_parameters', 'deepcci_net'))
    axes = plt.gca()
    axes.set(xlabel='epoch', ylabel='accuracy')
    axes.grid()
    plt.savefig(os.path.join(results_path, plot_name), dpi=600)
    """
    from learning.utils import *
    result_path="/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/Thesis/my_net_vs_deepcci_net/60seconds_0_background_flows/"
    create_accuracy_vs_epoch_graph(result_path,"validation_accuracy","Accuracy VS Session Duration (0 background flows)")
    """

def create_test_only_graph(results_path, txt_filename, plot_name):
    my_net_all_parameters_accuracy_list = []
    for dir_name in os.listdir(results_path):
        res_dir = os.path.join(results_path, dir_name)
        if not os.path.isdir(res_dir):
            continue
        x_axis = re.findall(r'\d+', dir_name)
        res_file = os.path.join(res_dir, txt_filename)
        with open(res_file) as f:
            accuracy = f.readlines()
            accuracy = [x.strip() for x in accuracy]
            my_net_all_parameters_accuracy_list.append((int(x_axis[0]), float(accuracy[1])))
    plt.cla()  # clear the current axes
    plt.clf()  # clear the current figure
    my_net_all_parameters_accuracy_list = sorted(my_net_all_parameters_accuracy_list, key=lambda tup: tup[0])
    x_axis = [x[0] for x in my_net_all_parameters_accuracy_list]
    y_axis = [x[1] for x in my_net_all_parameters_accuracy_list]
    plt.plot(x_axis, y_axis)
    axes = plt.gca()
    axes.set(xlabel='number of flows', ylabel='F1')
    axes.grid()
    plt.title(plot_name)
    plt.savefig(os.path.join(results_path, plot_name), dpi=600)
    """
    from learning.utils import *
    result_path="/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/Thesis/test_only_scalability/f1_all_parameters_model/10seconds_75_background_flows/"
    create_test_only_graph(result_path,"validation_f1","Accuracy VS Number of Background Flows")
    """

def get_test_only_graph(results_path, txt_filename, plot_name, is_scatter = False):
    my_net_all_parameters_accuracy_list = []
    for dir_name in os.listdir(results_path):
        res_dir = os.path.join(results_path, dir_name)
        if not os.path.isdir(res_dir):
            continue
        if "model" in res_dir or "old" in res_dir:
            continue
        x_axis = re.findall(r'\d+', dir_name)
        res_file = os.path.join(res_dir, txt_filename)
        with open(res_file) as f:
            accuracy = f.readlines()
            accuracy = [x.strip() for x in accuracy]
            if is_scatter:
                my_net_all_parameters_accuracy_list.append((int(x_axis[0]), float(accuracy[-1])))
            else:
                my_net_all_parameters_accuracy_list.append((int(x_axis[0]), float(accuracy[1])))
    return my_net_all_parameters_accuracy_list

def create_test_only_graphs(results_path, txt_filename, plot_name):
    my_net_all_parameters_accuracy_list = []
    # graph_legend = ["10seconds", "20seconds", "60seconds"]
    graph_legend = ["all_parameters", "cbiq", "deepcci", "throughput"]
    graph_legend_aligned = []
    for dir_name in os.listdir(results_path):
        res_dir = os.path.join(results_path, dir_name)
        if not os.path.isdir(res_dir):
            continue
        for res_sub_dir in os.listdir(os.path.join(results_path, dir_name, res_dir)):
            if not os.path.isdir(res_dir) or "20" not in res_sub_dir:
                continue
            for sub_dir in os.listdir(os.path.join(results_path, dir_name, res_sub_dir)):
                if "model" in sub_dir:
                    continue
                for graph_type in graph_legend:
                    if graph_type in res_dir:
                        graph_legend_aligned.append(graph_type)
                result_path = os.path.join(results_path, dir_name, res_dir, res_sub_dir, sub_dir)
                my_net_all_parameters_accuracy_list.append(get_test_only_graph(result_path, txt_filename, plot_name))
    plt.cla()  # clear the current axes
    plt.clf()  # clear the current figure
    for i in range(len(my_net_all_parameters_accuracy_list)):
        my_net_all_parameters_accuracy = sorted(my_net_all_parameters_accuracy_list[i], key=lambda tup: tup[0])
        x_axis = [x[0] for x in my_net_all_parameters_accuracy]
        y_axis = [x[1] for x in my_net_all_parameters_accuracy]
        plt.plot(x_axis, y_axis)
    axes = plt.gca()
    axes.set(xlabel='number of flows', ylabel='F1')
    axes.grid()
    plt.legend(graph_legend_aligned)
    plt.title(plot_name)
    plt.savefig(os.path.join(results_path, plot_name), dpi=600)

def create_test_only_diverse_graphs(results_path, txt_filename, plot_name):
    my_net_all_parameters_accuracy_list = []
    my_net_all_parameters_scatter = []
    graph_legend = ["diverse", "0_background_flows", "75_background_flows"]
    graph_legend_aligned = []
    for dir_name in os.listdir(results_path):
        res_dir = os.path.join(results_path, dir_name)
        if not os.path.isdir(res_dir):
            continue
        if "points" in res_dir:
            my_net_all_parameters_scatter.append(get_test_only_graph(os.path.join(results_path, dir_name), txt_filename, plot_name, True))
        for graph_type in graph_legend:
            if graph_type in dir_name:
                graph_legend_aligned.append(graph_type)
        result_path = os.path.join(results_path, dir_name, res_dir)
        my_net_all_parameters_accuracy_list.append(get_test_only_graph(result_path, txt_filename, plot_name))
    plt.cla()  # clear the current axes
    plt.clf()  # clear the current figure
    for i in range(len(my_net_all_parameters_accuracy_list)):
        my_net_all_parameters_accuracy = sorted(my_net_all_parameters_accuracy_list[i], key=lambda tup: tup[0])
        x_axis = [x[0] for x in my_net_all_parameters_accuracy]
        y_axis = [x[1] for x in my_net_all_parameters_accuracy]
        plt.plot(x_axis, y_axis)
    for i in range(len(my_net_all_parameters_scatter)):
        my_net_all_parameters_accuracy = sorted(my_net_all_parameters_scatter[i], key=lambda tup: tup[0])
        x_axis = [x[0] for x in my_net_all_parameters_accuracy]
        y_axis = [x[1] for x in my_net_all_parameters_accuracy]
        plt.scatter(x_axis, y_axis)
    axes = plt.gca()
    axes.set(xlabel='number of flows', ylabel='F1')
    axes.grid()
    plt.legend(graph_legend_aligned)
    plt.title(plot_name)
    plt.savefig(os.path.join(results_path, plot_name), dpi=600)


def get_f1_vs_background_flows(results_path, txt_filename, plot_name):
    my_net_all_parameters_accuracy_list = []
    for dir_name in os.listdir(results_path):
        res_dir = os.path.join(results_path, dir_name)
        if not os.path.isdir(res_dir):
            continue
        x_axis = re.findall(r'\d+', dir_name)
        res_file = os.path.join(res_dir, txt_filename)
        try:
            with open(res_file) as f:
                accuracy = f.readlines()
                my_net_all_parameters_accuracy_list.append((int(x_axis[0]), float(accuracy[-1])))
        except:
            continue
    return my_net_all_parameters_accuracy_list

def create_f1_vs_background_flows_multiple_session_duration_graph(results_path, txt_filename, plot_name, session_duration):
    my_net_all_parameters_accuracy_list = []
    graph_legend = ["all_parameters", "cbiq", "deepcci", "throughput"]
    graph_legend_aligned = []
    for dir_name in os.listdir(results_path):
        res_dir = os.path.join(results_path, dir_name)
        if not os.path.isdir(res_dir) or "old" in res_dir or "multiple_rtr" in res_dir:
            continue
        for sub_dir in os.listdir(os.path.join(results_path, dir_name, res_dir)):
            if session_duration not in sub_dir:
                continue
            for graph_type in graph_legend:
                if graph_type in res_dir:
                    graph_legend_aligned.append(graph_type)
            result_path = os.path.join(results_path, dir_name, res_dir, sub_dir)
            my_net_all_parameters_accuracy_list.append(get_f1_vs_background_flows(result_path, txt_filename, plot_name))
    plt.cla()  # clear the current axes
    plt.clf()  # clear the current figure
    for i in range(len(my_net_all_parameters_accuracy_list)):
        my_net_all_parameters_accuracy = sorted(my_net_all_parameters_accuracy_list[i], key=lambda tup: tup[0])
        x_axis = [x[0] for x in my_net_all_parameters_accuracy]
        y_axis = [x[1] for x in my_net_all_parameters_accuracy]
        plt.plot(x_axis, y_axis)
    axes = plt.gca()
    axes.set(xlabel='number of flows', ylabel='F1')
    axes.grid()
    plt.title(plot_name)
    plt.legend(graph_legend_aligned)
    plt.savefig(os.path.join(results_path, plot_name), dpi=600)

def create_multiple_rtr_graph(results_path, txt_filename, plot_name, session_duration):
    f1_list = []
    graph_legend = ["all_parameters", "cbiq", "deepcci", "throughput"]
    graph_legend_aligned = []
    for dir_name in os.listdir(results_path):
        res_dir = os.path.join(results_path, dir_name)
        if not os.path.isdir(res_dir) or "old" in res_dir:
            continue
        for sub_dir in os.listdir(os.path.join(results_path, dir_name, res_dir)):
            result_path = os.path.join(results_path, dir_name, res_dir, sub_dir)
            if not os.path.isdir(result_path) or "old" in result_path:
                continue
            for graph_type in graph_legend:
                if graph_type in sub_dir:
                    graph_legend_aligned.append(graph_type)
            f1_list.append(get_f1_vs_background_flows(result_path, txt_filename, plot_name))
    plt.cla()  # clear the current axes
    plt.clf()  # clear the current figure
    rtr1 = [x[0][1] for x in f1_list]
    rtr2 = [x[1][1] for x in f1_list]

    plt.figure(figsize=(10, 5))
    ind = np.arange(len(rtr1))
    width = 0.3
    plt.bar(ind, rtr1, width)
    plt.bar(ind + width, rtr2, width)
    plt.xticks(ind + width / 2, graph_legend_aligned)

    axes = plt.gca()
    axes.set(xlabel='Datasets', ylabel='F1')
    axes.grid()
    plt.title(plot_name)
    plt.legend(("rtr1", "rtr2"), loc="best")
    plt.savefig(os.path.join(results_path, plot_name), dpi=600)

    """
    from learning.utils import *
    result_path="/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/Thesis/new_topology/multiple_rtr/30_background_flows/"
    create_multiple_rtr_graph(result_path,"validation_accuracy","F1 for 30 Background Flows", "20")
    """

class Graph_Creator:
    def __init__(self, loss, accuracy, accuracy_per_type, num_of_epochs, is_batch, plot_file_name="Graph.png", plot_fig_name="Statistics"):
        #loss = np.array(loss, dtype=np.float32)
        #accuracy = np.array(accuracy, dtype=np.float32)
        # clear plt before starting new statistics, otherwise it add up to the previous one
        self.is_batch = is_batch
        self.loss = loss
        self.accuracy = accuracy
        self.accuracy_per_type = accuracy_per_type
        plt.cla()  # clear the current axes
        plt.clf()  # clear the current figure
        fig1, (ax1, ax2, ax3) = plt.subplots(3, constrained_layout=True)
        fig1.suptitle(plot_fig_name)
        ax1.set(xlabel='epoch', ylabel='loss')
        ax1.set_xlim([0, num_of_epochs])
        #ax1.set_ylim([0, numpy.amax(loss)])
        ax2.set(xlabel='epoch', ylabel='accuracy')
        ax2.set_xlim([0, num_of_epochs])
        #ax2.set_ylim([0, numpy.amax(accuracy)])
        self.fig1 = fig1
        self.loss_ax = ax1
        self.accuracy_ax = ax2
        self.accuracy_per_type_ax = ax3
        if self.is_batch:
            self.average_loss = numpy.mean(loss, axis=1)
            self.average_accuracy = numpy.mean(accuracy, axis=1)
            self.average_accuracy_per_type = numpy.mean(accuracy_per_type, axis=1)
            fig2, (ax4, ax5) = plt.subplots(2, constrained_layout=True)
            fig2.suptitle(plot_fig_name)
            ax4.set(xlabel='epoch', ylabel='loss')
            ax4.set_xlim([0, num_of_epochs])
            #ax4.set_ylim([0, numpy.amax(loss)])
            ax5.set(xlabel='epoch', ylabel='accuracy')
            ax5.set_xlim([0, num_of_epochs])
            #ax5.set_ylim([0, numpy.amax(accuracy)])
            self.fig2 = fig2
            self.batches_loss_ax = ax4
            self.batches_accuracy_ax = ax5
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

    def create_accuracy_per_type_plot(self, accuracy_per_type, ax):
        accuracy_per_type_df = pd.DataFrame(accuracy_per_type)
        for type in range(accuracy_per_type_df.shape[1]):
            accuracy_per_type_df[type].plot(kind='line', ax=ax, title='accuracy per type')
        ax.legend(["bbr", "cubic", "reno"])
        self.write_to_file("accuracy_per_type", accuracy_per_type)

    def create_graphs(self):
        if self.is_batch:
            self.create_loss_plot(self.average_loss, self.loss_ax)
            self.create_accuracy_plot(self.average_accuracy, self.accuracy_ax)
            self.create_accuracy_per_type_plot(self.average_accuracy_per_type, self.accuracy_per_type_ax)
            self.loss_ax.grid()
            self.accuracy_ax.grid()
            self.accuracy_per_type_ax.grid()
            self.save_and_show(self.fig1)
            self.plot_file_name = self.plot_file_name.replace('.png', ('_batches_figures.png'))
            self.create_loss_plot(self.loss, self.batches_loss_ax)
            self.create_accuracy_plot(self.accuracy, self.batches_accuracy_ax)
            # self.create_accuracy_per_type_plot(self.accuracy_per_type, self.batches_accuracy_per_type_ax)
            self.batches_loss_ax.grid()
            self.batches_accuracy_ax.grid()
            self.save_and_show(self.fig2)
        else:
            self.create_loss_plot(self.loss, self.loss_ax)
            self.create_accuracy_plot(self.accuracy, self.accuracy_ax)
            self.create_accuracy_per_type_plot(self.average_accuracy_per_type, self.accuracy_per_type_ax)
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
