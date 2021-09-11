import os
import torch
import torch.utils as torch_utils
import numpy
import re
import statistics
import matplotlib.pyplot as plt
# import project functions
from learning.env import *
from learning.results_manager import *
from learning.env import *


# Results1:
"""
from learning.thesis_graphs_utils import *
result_path = "/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/thesis_prime/Accuracy VS Epochs- Results1- Results1/"
create_accuracy_vs_epoch_graph(result_path, "validation_accuracy", "Results1", True)
"""
def create_accuracy_vs_epoch_graph(results_path, txt_filename, plot_name, short_epoch):
    my_net_accuracy = []
    my_net_all_parameters_accuracy = []
    deepcci_net_accuracy = []
    for dir_name in os.listdir(results_path):
        res_dir = os.path.join(results_path, dir_name)
        if not os.path.isdir(res_dir):
            continue
        res_file = os.path.join(res_dir, txt_filename)
        with open(res_file) as f:
            accuracy = f.readlines()
            accuracy = [x.strip() for x in accuracy]
            accuracy = [float(i) for i in accuracy]
            if "5parameters" in res_file:
                my_net_all_parameters_accuracy = accuracy
            else:
                if "cbiq" in res_file:
                    my_net_accuracy = accuracy
                else:
                    deepcci_net_accuracy = accuracy
    plt.cla()  # clear the current axes
    plt.clf()  # clear the current figure
    if short_epoch:
        p1, = plt.plot(my_net_accuracy[0:45])
        p2, = plt.plot(my_net_all_parameters_accuracy[0:45])
        p3, = plt.plot(deepcci_net_accuracy[0:45])
    else:
        p1, = plt.plot(my_net_accuracy)
        p2, = plt.plot(my_net_all_parameters_accuracy)
        p3, = plt.plot(deepcci_net_accuracy)
    plt.legend((p1, p2, p3), ('CBIQ', 'All Parameters', 'Deepcci'))
    axes = plt.gca()
    axes.set(xlabel='epoch', ylabel='f1')
    axes.grid()
    plt.savefig(os.path.join(results_path, plot_name), dpi=600)


# Results2:
"""
from learning.thesis_graphs_utils import *
result_path = "/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/thesis_prime/Accuracy VS Session Duration (15 background tcp flows)- Results2"
create_acuuracy_vs_session_duration_graph(result_path, "validation_accuracy", "accuracy_vs_session_duration")
"""
def create_f1_vs_session_duration_graph(results_path, txt_filename, plot_name):
    my_net_accuracy_list, my_net_all_parameters_accuracy_list, deepcci_net_accuracy_list = get_f1_vs_session_duration(
        results_path, txt_filename, plot_name, None)
    plt.cla()  # clear the current axes
    plt.clf()  # clear the current figure
    my_net_accuracy_list = sorted(my_net_accuracy_list, key=lambda tup: tup[1])
    session_duration = [x[1] for x in my_net_accuracy_list]
    accuracy = [x[0] for x in my_net_accuracy_list]
    p1, = plt.plot(session_duration, accuracy)
    # plt.title(plot_name)
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
        plt.legend((p1, p2, p3), ('CBIQ', 'All Parameters', 'Deepcci'))
    else:
        if p3 is not None:
            plt.legend((p1, p3), ('CBIQ', 'Deepcci'))
    axes = plt.gca()
    axes.set(xlabel='session duration[seconds]', ylabel='f1')
    axes.grid()
    plt.savefig(os.path.join(results_path, plot_name), dpi=600)


# Results3:
"""
from learning.thesis_graphs_utils import *
result_path = "/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/thesis_prime/f1 vs number of flows (10 seconds)- Results3/"
create_f1_vs_number_of_flows_graph(result_path, "validation_accuracy", "Results3", 10000)
"""
def create_f1_vs_number_of_flows_graph(results_path, txt_filename, plot_name, session_duration):
    my_net_accuracy_list = []
    my_net_all_parameters_accuracy_list = []
    deepcci_net_accuracy_list = []
    for dir_name in os.listdir(results_path):
        number_of_flows = re.findall(r'\d+', dir_name)
        res_dir = os.path.join(results_path, dir_name)
        if not os.path.isdir(res_dir):
            continue
        my_net_accuracy, my_net_all_parameters_accuracy, deepcci_net_accuracy = get_f1_vs_nubmber_of_flows(
            res_dir, txt_filename, plot_name, session_duration)
        if my_net_accuracy:
            my_net_accuracy_list_accuracy = [x[0] for x in my_net_accuracy]
            my_net_accuracy_list.append((int(number_of_flows[0]), my_net_accuracy_list_accuracy[0]))
        if my_net_all_parameters_accuracy:
            my_net_all_parameters_accuracy_list_accuracy = [x[0] for x in my_net_all_parameters_accuracy]
            my_net_all_parameters_accuracy_list.append(
                (int(number_of_flows[0]), my_net_all_parameters_accuracy_list_accuracy[0]))
        if deepcci_net_accuracy:
            deepcci_net_accuracy_list_accuracy = [x[0] for x in deepcci_net_accuracy]
            deepcci_net_accuracy_list.append((int(number_of_flows[0]), deepcci_net_accuracy_list_accuracy[0]))
    plt.cla()  # clear the current axes
    plt.clf()  # clear the current figure
    # plt.title(plot_name)
    my_net_accuracy_list = sorted(my_net_accuracy_list, key=lambda tup: tup[0])
    number_of_flows = [x[0] for x in my_net_accuracy_list]
    accuracy = [x[1] for x in my_net_accuracy_list]
    p1, = plt.plot(number_of_flows, accuracy)
    p2 = p3 = None
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
        plt.legend((p1, p2, p3), ('CBIQ', 'All Parameters', 'Deepcci'))
    else:
        if p3 is not None:
            plt.legend((p1, p3), ('CBIQ', 'Deepcci'))
    axes = plt.gca()
    axes.set(xlabel='number of flows', ylabel='f1')
    # axes.set_xlim([0, len(my_net_accuracy_list)])
    # plt.xticks(number_of_flows)
    # axes.set_ylim([0, numpy.amax(my_net_accuracy_list)])
    axes.grid()
    plt.savefig(os.path.join(results_path, plot_name), dpi=600)


# Results10:
"""
from learning.thesis_graphs_utils import *
result_path="/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/thesis_prime/new topology- f1 vs number of background flows (20 seconds)- Results10/"
create_f1_vs_background_flows_multiple_session_duration_graph(result_path,"validation_accuracy","f1 for 15 Background Flows", "20")
"""
def create_f1_vs_background_flows_multiple_session_duration_graph(results_path, txt_filename, plot_name, session_duration):
    my_net_all_parameters_accuracy_list = []
    graph_legend = ["All Parameters", "CBIQ", "Deepcci", "Throughput"]
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
            my_net_all_parameters_accuracy_list.append(get_f1_result_for_subfolders_from_accuracy(result_path, txt_filename, plot_name))
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
    plt.savefig(os.path.join(results_path, plot_name), dpi=600)


# Results11:
"""
from learning.thesis_graphs_utils import *
result_path="/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/thesis_prime/new topology- classification of different datasets- in multiple routers- Results11/15_background_flows/20seconds/"
create_multiple_rtr_graph(result_path,"validation_accuracy","f1 for 15 Background Flows")
"""
def create_multiple_rtr_graph(results_path, txt_filename, plot_name):
    f1_list = []
    f1_sublist = []
    graph_legend = ["All Parameters", "CBIQ", "Deepcci", "Throughput"]
    graph_legend_aligned = []
    for dir_name in os.listdir(results_path):
        res_dir = os.path.join(results_path, dir_name)
        if not os.path.isdir(res_dir) or "old" in res_dir:
            continue
        for sub_dir in os.listdir(os.path.join(results_path, dir_name, res_dir)):
            result_path = os.path.join(results_path, dir_name, res_dir, sub_dir)
            if not os.path.isdir(result_path) or "old" in result_path:
                continue
            f1_sublist.append(get_f1_result_multiple_rtr(result_path, txt_filename))
        f1_list.append(f1_sublist)
        f1_sublist = []
        for graph_type in graph_legend:
            if graph_type in dir_name:
                graph_legend_aligned.append(graph_type)
    plt.cla()  # clear the current axes
    plt.clf()  # clear the current figure
    rtr1 = [x[0][0] for x in f1_list]
    rtr2 = [x[1][0] for x in f1_list]
    plt.figure(figsize=(10, 5))
    ind = np.arange(len(rtr1))
    width = 0.3
    plt.bar(ind, rtr1, width)
    plt.bar(ind + width, rtr2, width)
    plt.xticks(ind + width / 2, graph_legend_aligned)
    axes = plt.gca()
    axes.set(xlabel='Datasets', ylabel='F1')
    axes.grid()
    plt.legend(("rtr2", "rtr1"), loc="best")
    plt.savefig(os.path.join(results_path, plot_name), dpi=600)


# Results12:
"""
from learning.thesis_graphs_utils import *
result_path="/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/thesis_prime/pre- trained model- f1 vs number of background flows (different session duration)- Results12/"
create_f1_vs_background_flows_for_pre_trained_model_and_session_durations(result_path,"validation_f1","f1 vs number of background flows (different session duration)")
"""
def create_f1_vs_background_flows_for_pre_trained_model_and_session_durations(results_path, txt_filename, plot_name):
    my_net_all_parameters_accuracy_list = []
    graph_legend = ["10 seconds", "20 seconds", "60 seconds"]
    graph_legend_aligned = []
    for dir_name in os.listdir(results_path):
        res_dir = os.path.join(results_path, dir_name)
        if not os.path.isdir(res_dir):
            continue
        for sub_dir_name in os.listdir(res_dir):
            sub_res_dir = os.path.join(results_path, dir_name, sub_dir_name)
            if not os.path.isdir(sub_res_dir) or "_model" in sub_res_dir:
                continue
            for graph_type in graph_legend:
                if graph_type in sub_res_dir:
                    graph_legend_aligned.append(graph_type)
            result_path = os.path.join(results_path, dir_name, res_dir, sub_res_dir)
            my_net_all_parameters_accuracy_list.append(get_pre_trained_model_result(result_path, txt_filename, plot_name))
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
    plt.savefig(os.path.join(results_path, plot_name), dpi=600)


# Results13:
"""
from learning.thesis_graphs_utils import *
result_path="/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/thesis_prime/pre- trained model- f1 vs number of background flows for dirverse training (multiple rtr, 20 seconds)- Results13/20seconds/"
create_f1_vs_number_of_flows(result_path,"validation_f1","f1 vs number of background flows (20 seconds)")
"""
def create_f1_vs_number_of_flows(results_path, txt_filename, plot_name):
    my_net_all_parameters_accuracy_list = []
    graph_legend = ["All Parameters", "CBIQ", "Deepcci", "Throughput"]
    graph_legend_aligned = []
    for dir_name in os.listdir(results_path):
        res_dir = os.path.join(results_path, dir_name)
        if not os.path.isdir(res_dir) or "Throughput" in res_dir:
            continue
        for res_sub_dir in os.listdir(os.path.join(results_path, dir_name, res_dir)):
            if "model" in res_sub_dir:
                continue
            for graph_type in graph_legend:
                if graph_type in res_dir:
                    graph_legend_aligned.append(graph_type)
            result_path = os.path.join(results_path, dir_name, res_dir, res_sub_dir)
            my_net_all_parameters_accuracy_list.append(get_pre_trained_model_result(result_path, txt_filename, plot_name))
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
    # plt.title(plot_name)
    plt.savefig(os.path.join(results_path, plot_name), dpi=600)

# Results14:
"""
from learning.thesis_graphs_utils import *
result_path="/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/thesis_prime/pre- trained model- f1 vs number of background flows (20 seconds)- Results14/All_Parameters/20seconds/"
create_f1_vs_background_flows_for_pre_trained_model(result_path,"validation_f1","f1 vs number of background flows (20 seconds)")
"""
def create_f1_vs_background_flows_for_pre_trained_model(results_path, txt_filename, plot_name):
    my_net_all_parameters_accuracy_list = []
    my_net_all_parameters_scatter = []
    graph_legend = ["Diverse Training Pre- Trained Model", "0 Background Flows Pre- Trained Model", "75 Background Flows Pre- Trained Model"]
    scatter_legend = ["0", "15", "30", "75"]
    graph_legend_aligned = []
    scatter_legend_aligned = []
    for dir_name in os.listdir(results_path):
        res_dir = os.path.join(results_path, dir_name)
        if not os.path.isdir(res_dir):
            continue
        if "points" in res_dir:
            for scatter_type in scatter_legend:
                if scatter_type in dir_name:
                    scatter_legend_aligned.append(scatter_type)
            if "added_points" in res_dir:
                my_net_all_parameters_scatter.append(get_pre_trained_model_result(os.path.join(results_path, dir_name), txt_filename, plot_name, True))
            else:
                my_net_all_parameters_scatter.append(get_pre_trained_model_result(os.path.join(results_path, dir_name), txt_filename, plot_name, False))
        else:
            for graph_type in graph_legend:
                if graph_type in dir_name:
                    graph_legend_aligned.append(graph_type)
            result_path = os.path.join(results_path, dir_name, res_dir)
            my_net_all_parameters_accuracy_list.append(get_pre_trained_model_result(result_path, txt_filename, plot_name))
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
        plt.scatter(x_axis, y_axis, color = "red")
    axes = plt.gca()
    axes.set(xlabel='number of flows', ylabel='F1')
    axes.grid()
    plt.legend(graph_legend_aligned)#, loc=(0.75,0.5))
    plt.savefig(os.path.join(results_path, plot_name), dpi=600)


# Results15:
"""
from learning.thesis_graphs_utils import *
result_path="/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/thesis_prime/classification of different datasets using a single trained model- in multiple routers- Results15/10 seconds, 75 background flows/"
create_diverse_multiple_rtr_graph(result_path,"validation_f1","f1 for 75 Background Flows", "10")
"""
def create_diverse_multiple_rtr_graph(results_path, txt_filename, plot_name, session_duration):
    f1_list = []
    graph_legend = ["5parameters", "cbiq", "deepcci", "throughput"]
    graph_legend_aligned = []
    for dir_name in os.listdir(results_path):
        res_dir = os.path.join(results_path, dir_name)
        if not os.path.isdir(res_dir) or "old" in res_dir:
            continue
        for graph_type in graph_legend:
            if graph_type in dir_name:
                graph_legend_aligned.append(graph_type)
        f1_list.append(get_f1_result_for_subfolders(res_dir, txt_filename, plot_name))
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
    plt.legend(("rtr2", "rtr1"), loc="best")
    plt.savefig(os.path.join(results_path, plot_name), dpi=600)


# Results16:
"""
from learning.thesis_graphs_utils import *
result_path="/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/thesis_prime/f1- for each parameter seperately/10 seconds with 75 background flows/"
create_result_for_each_parameter_graph(result_path,"validation_accuracy","f1 for 75 Background Flows")
"""
def create_result_for_each_parameter_graph(results_path, txt_filename, plot_name):
    f1_list = []
    graph_legend = ["CBIQ", "In Throughput", "Out Throughput", "Number of Drops", "Number of Retransmits", "Send Time Gap"]
    graph_legend_aligned = []
    for dir_name in os.listdir(results_path):
        res_dir = os.path.join(results_path, dir_name)
        if not os.path.isdir(res_dir) or "old" in res_dir:
            continue
        for graph_type in graph_legend:
            if graph_type in dir_name:
                graph_legend_aligned.append(graph_type)
        f1_list.append(get_f1_result(res_dir, txt_filename))
    plt.cla()  # clear the current axes
    plt.clf()  # clear the current figure
    f1 = [x[0][1] for x in f1_list]

    plt.figure(figsize=(10, 5))
    ind = np.arange(len(f1))
    width = 0.3
    plt.bar(ind, f1, width)
    plt.xticks(ind + width / 2, graph_legend_aligned)

    axes = plt.gca()
    axes.set(xlabel='Parameter', ylabel='F1')
    axes.grid()
    plt.savefig(os.path.join(results_path, plot_name), dpi=600)


# Results17:
"""
from learning.thesis_graphs_utils import *
result_path="/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/thesis_prime/networks comparison- Results17/10 seconds with 75 background flows/"
create_networks_comparison_graph(result_path,"validation_accuracy","f1 for 75 Background Flows")
"""
def create_networks_comparison_graph(results_path, txt_filename, plot_name):
    f1_list = []
    graph_legend = ["Fully Connected", "Deepcci", "CNN"]
    graph_legend_aligned = []
    for dir_name in os.listdir(results_path):
        res_dir = os.path.join(results_path, dir_name)
        if not os.path.isdir(res_dir) or "old" in res_dir:
            continue
        for graph_type in graph_legend:
            if graph_type in dir_name:
                graph_legend_aligned.append(graph_type)
        f1_list.append(get_f1_result(res_dir, txt_filename))
    plt.cla()  # clear the current axes
    plt.clf()  # clear the current figure
    plt.figure(figsize=(10, 5))
    f1 = [x[0][1] for x in f1_list]
    ind = np.arange(len(f1))
    width = 0.3
    plt.bar(ind, f1, width)
    plt.xticks(ind, graph_legend_aligned)
    axes = plt.gca()
    axes.set(xlabel='Parameter', ylabel='F1')
    axes.grid()
    plt.savefig(os.path.join(results_path, plot_name), dpi=600)


# Results18:
"""
from learning.thesis_graphs_utils import *
result_path="/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/thesis_prime/online_classification/networks comparison/9499 chunk size/background flows/30 background flows/"
create_online_classification_graph(result_path,"validation_accuracy","f1 for 30 Background Flows")
"""
def create_online_classification_graph(results_path, txt_filename, plot_name):
    f1_list = []
    graph_legend = ["CBIQ", "Throughput", "All Parameters", "Deepcci"]
    graph_legend_aligned = []
    for dir_name in os.listdir(results_path):
        res_dir = os.path.join(results_path, dir_name)
        if not os.path.isdir(res_dir) or "old" in res_dir:
            continue
        for graph_type in graph_legend:
            if graph_type in dir_name:
                graph_legend_aligned.append(graph_type)
        f1_list.append(get_f1_result(res_dir, txt_filename))
    plt.cla()  # clear the current axes
    plt.clf()  # clear the current figure
    f1 = [x[0][1] for x in f1_list]

    plt.figure(figsize=(10, 5))
    ind = np.arange(len(f1))
    width = 0.3
    plt.bar(ind, f1, width)
    plt.xticks(ind, graph_legend_aligned)

    axes = plt.gca()
    axes.set(xlabel='Parameter', ylabel='F1')
    axes.grid()
    plt.savefig(os.path.join(results_path, plot_name), dpi=600)


# Results19:
"""
from learning.thesis_graphs_utils import *
result_path="/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/thesis_prime/online_classification/networks comparison/9499 chunk size/parameters/"
create_f1_vs_background_flows_for_online_classification_graph(result_path,"validation_accuracy","f1 vs number of background flows (10 seconds)")
"""
def create_f1_vs_background_flows_for_online_classification_graph(results_path, txt_filename, plot_name):
    my_net_all_parameters_accuracy_list = []
    my_net_all_parameters_scatter = []
    graph_legend = ["Deepcci", "CBIQ", "Throughput"]#,"All Parameters"]
    graph_legend_aligned = []
    for dir_name in os.listdir(results_path):
        res_dir = os.path.join(results_path, dir_name)
        if not os.path.isdir(res_dir) or "old" in res_dir:
            continue
        for graph_type in graph_legend:
            if graph_type in dir_name:
                graph_legend_aligned.append(graph_type)
        result_path = os.path.join(results_path, dir_name, res_dir)
        my_net_all_parameters_accuracy_list.append(get_f1_result_for_subfolders_from_online_accuracy(result_path, txt_filename))
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
        plt.scatter(x_axis, y_axis, color = "red")
    axes = plt.gca()
    axes.set(xlabel='number of flows', ylabel='F1')
    axes.grid()
    plt.legend(graph_legend_aligned)#, loc=(0.75,0.5))
    plt.savefig(os.path.join(results_path, plot_name), dpi=600)


# Results20:
"""
from learning.thesis_graphs_utils import *
result_path="/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/thesis_prime/online_classification/sampling rate/9499 chunk size/background flows/0 background flows"
create_f1_vs_background_flows_for_online_classification_with_sampling_graph(result_path,"validation_accuracy","f1 vs number of background flows")
"""
def create_f1_vs_background_flows_for_online_classification_with_sampling_graph(results_path, txt_filename, plot_name):
    f1_list = []
    graph_legend = ["CBIQ", "Throughput", "All Parameters", "Deepcci"]
    graph_legend_aligned = []
    for dir_name in os.listdir(results_path):
        res_dir = os.path.join(results_path, dir_name)
        if not os.path.isdir(res_dir) or "old" in res_dir:
            continue
        for graph_type in graph_legend:
            if graph_type in dir_name:
                graph_legend_aligned.append(graph_type)
        f1_list.append(get_f1_result_for_sampling_rate(res_dir, txt_filename))
    plt.cla()  # clear the current axes
    plt.clf()  # clear the current figure
    f1 = [x[0][1] for x in f1_list]

    plt.figure(figsize=(10, 5))
    ind = np.arange(len(f1))
    width = 0.3
    plt.bar(ind, f1, width)
    plt.xticks(ind, graph_legend_aligned)

    axes = plt.gca()
    axes.set(xlabel='Parameter', ylabel='F1')
    axes.grid()
    plt.savefig(os.path.join(results_path, plot_name), dpi=600)



# Results21:
"""
from learning.thesis_graphs_utils import *
result_path="/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/thesis_prime/online_classification/sampling rate/9499 chunk size/every tenth packet- global sampling/parameters/"
create_f1_vs_parameter_for_online_classification_with_sampling_graph(result_path,"validation_accuracy","f1 vs number of background flows")
"""
def create_f1_vs_parameter_for_online_classification_with_sampling_graph(results_path, txt_filename, plot_name):
    my_net_all_parameters_accuracy_list = []
    graph_legend = ["Deepcci", "CBIQ", "Throughput", "All Parameters"]
    graph_legend_aligned = []
    for dir_name in os.listdir(results_path):
        res_dir = os.path.join(results_path, dir_name)
        if not os.path.isdir(res_dir) or "old" in res_dir:
            continue
        for graph_type in graph_legend:
            if graph_type in dir_name:
                graph_legend_aligned.append(graph_type)
        result_path = os.path.join(results_path, dir_name, res_dir)
        my_net_all_parameters_accuracy_list.append(get_f1_result_for_subfolders_from_online_accuracy(result_path, txt_filename))
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
    plt.legend(graph_legend_aligned)#, loc=(0.75,0.5))
    plt.savefig(os.path.join(results_path, plot_name), dpi=600)


def get_f1_vs_session_duration(results_path, txt_filename, plot_name, single_session_duration):
    my_net_accuracy_list = []
    my_net_all_parameters_accuracy_list = []
    deepcci_net_accuracy_list = []
    for dir_name in os.listdir(results_path):
        if "old" in dir_name:
            continue
        session_duration = re.findall(r'\d+', dir_name)
        res_dir = os.path.join(results_path, dir_name)
        if not os.path.isdir(res_dir):
            continue
        if single_session_duration is not None:
            if len(re.findall(str(single_session_duration) + '_', res_dir)) == 0:
                continue
        res_file = os.path.join(res_dir, txt_filename)
        try:
            with open(res_file) as f:
                accuracy = f.readlines()
                accuracy = [x.strip() for x in accuracy]
                accuracy = [float(i) for i in accuracy]
                if "my_net" in res_file:
                    if "5parameters" in res_file:
                            my_net_all_parameters_accuracy_list.append(
                                (statistics.mean(accuracy[80:]), int(session_duration[1]) / 1000))
                    else:
                        my_net_accuracy_list.append((statistics.mean(accuracy[80:]), int(session_duration[0]) / 1000))
                else:
                    deepcci_net_accuracy_list.append((accuracy[1], int(session_duration[0]) / 1000))
        except:
            continue
    return my_net_accuracy_list, my_net_all_parameters_accuracy_list, deepcci_net_accuracy_list



def get_f1_vs_nubmber_of_flows(results_path, txt_filename, plot_name, single_session_duration):
    my_net_accuracy_list = []
    my_net_all_parameters_accuracy_list = []
    deepcci_net_accuracy_list = []
    for dir_name in os.listdir(results_path):
        if "old" in dir_name:
            continue
        session_duration = re.findall(r'\d+', dir_name)
        res_dir = os.path.join(results_path, dir_name)
        if not os.path.isdir(res_dir):
            continue
        if single_session_duration is not None:
            if len(re.findall(str(single_session_duration) + '_', res_dir)) == 0:
                continue
        res_file = os.path.join(res_dir, txt_filename)
        try:
            with open(res_file) as f:
                accuracy = f.readlines()
                accuracy = [x.strip() for x in accuracy]
                accuracy = [float(i) for i in accuracy]
                if "my_net" in res_file:
                    if "5parameters" in res_file:
                            my_net_all_parameters_accuracy_list.append(
                                (statistics.mean(accuracy[80:]), int(session_duration[1]) / 1000))
                    else:
                        my_net_accuracy_list.append((statistics.mean(accuracy[80:]), int(session_duration[0]) / 1000))
                else:
                    #deepcci_net_accuracy_list.append((accuracy[1], int(session_duration[0]) / 1000))
                    deepcci_net_accuracy_list.append((statistics.mean(accuracy[80:]), int(session_duration[0]) / 1000))
        except:
            continue
    return my_net_accuracy_list, my_net_all_parameters_accuracy_list, deepcci_net_accuracy_list



def get_f1_result(results_path, txt_filename):
    my_net_all_parameters_accuracy_list = []
    if not os.path.isdir(results_path):
        return
    x_axis = re.findall(r'\d+', results_path)
    res_file = os.path.join(results_path, txt_filename)
    try:
        with open(res_file) as f:
            accuracy = f.readlines()
            my_net_all_parameters_accuracy_list.append((int(x_axis[0]), float(accuracy[-1])))
    except:
        pass
    return my_net_all_parameters_accuracy_list


def get_f1_result_multiple_rtr(results_path, txt_filename):
    my_net_all_parameters_accuracy_list = []
    if not os.path.isdir(results_path):
        return
    res_file = os.path.join(results_path, txt_filename)
    try:
        with open(res_file) as f:
            accuracy = f.readlines()
            my_net_all_parameters_accuracy_list.append(float(accuracy[-1]))
    except:
        pass
    return my_net_all_parameters_accuracy_list


def get_f1_result_for_subfolders(results_path, txt_filename, plot_name):
    my_net_all_parameters_accuracy_list = []
    for dir_name in os.listdir(results_path):
        res_dir = os.path.join(results_path, dir_name)
        if not os.path.isdir(res_dir):
            continue
        if "old" in res_dir or "model" in dir_name:
            continue
        x_axis = re.findall(r'\d+', dir_name)
        res_file = os.path.join(res_dir, txt_filename)
        try:
            with open(res_file) as f:
                accuracy = f.readlines()
                my_net_all_parameters_accuracy_list.append((int(x_axis[0]), float(accuracy[1])))
        except:
            continue
    return my_net_all_parameters_accuracy_list


def get_f1_result_for_subfolders_from_accuracy(results_path, txt_filename, plot_name):
    my_net_all_parameters_accuracy_list = []
    for dir_name in os.listdir(results_path):
        res_dir = os.path.join(results_path, dir_name)
        if not os.path.isdir(res_dir):
            continue
        if "old" in res_dir or "model" in dir_name:
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


def get_pre_trained_model_result(results_path, txt_filename, plot_name, is_scatter = False):
    my_net_all_parameters_accuracy_list = []
    for dir_name in os.listdir(results_path):
        res_dir = os.path.join(results_path, dir_name)
        if not os.path.isdir(res_dir):
            continue
        if "model" in dir_name or "old" in dir_name:
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


def get_f1_result_for_subfolders_from_online_accuracy(results_path, txt_filename):
    my_net_all_parameters_accuracy_list = []
    for dir_name in os.listdir(results_path):
        res_dir = os.path.join(results_path, dir_name)
        if not os.path.isdir(res_dir):
            continue
        if "old" in res_dir or "model" in dir_name:
            continue
        x_axis = re.findall(r'\d+', res_dir)
        res_file = os.path.join(res_dir, txt_filename)
        try:
            with open(res_file) as f:
                accuracy = f.readlines()
                my_net_all_parameters_accuracy_list.append((int(x_axis[1]), float(accuracy[-1])))
        except:
            continue
    return my_net_all_parameters_accuracy_list


def get_f1_result_for_sampling_rate(results_path, txt_filename):
    my_net_all_parameters_accuracy_list = []
    if not os.path.isdir(results_path):
        return
    x_axis = re.findall(r'\d+', results_path)
    res_file = os.path.join(results_path, txt_filename)
    try:
        with open(res_file) as f:
            accuracy = f.readlines()
            my_net_all_parameters_accuracy_list.append((int(x_axis[0]), float(accuracy[-1])))
    except:
        pass
    return my_net_all_parameters_accuracy_list

# Online Filtering CBIQ debugging:
def get_online_filtered_cbiq(results_path, txt_filename):
    if not os.path.isdir(results_path):
        return
    from os import listdir
    from os.path import isfile, join
    dir_files = [f for f in listdir(results_path) if isfile(join(results_path, f))]
    for file in dir_files:
        if txt_filename in file:
            res_file = os.path.join(results_path, file)
            break
    try:
        df = pd.read_csv(res_file, index_col=None, header=0)
        return df['CBIQ']
    except:
        pass

"""
from learning.thesis_graphs_utils import *
result_path="/data_disk/cbiq_debugging"
create_online_filtered_cbiq_graphs(result_path,"random_sample_stat_bbr")
"""
def create_online_filtered_cbiq_graphs(results_path, txt_filename):
    cbiq_accuracy_lists = []
    graph_legend = ["in_ffill_out_ffill", "in_ffill_out_ffill_after_cbiq_calculation", "in_ffill_out_interpolation_after_cbiq_calculation", "in_interpolation_out_interpolation", "in_interpolation_out_interpolation_after_cbiq_calculation"]
    graph_legend_aligned = []
    for dir_name in os.listdir(results_path):
        res_dir = os.path.join(results_path, dir_name)
        if not os.path.isdir(res_dir) or "old" in res_dir:
            continue
        for graph_type in graph_legend:
            if graph_type in dir_name:
                graph_legend_aligned.append(graph_type)
        result_path = os.path.join(results_path, dir_name, res_dir)
        cbiq_accuracy_lists.append(get_online_filtered_cbiq(result_path, txt_filename))
    plt.cla()  # clear the current axes
    plt.clf()  # clear the current figure
    timestamp = list(range(0, 10000, 1))
    i = 1
    for cbiq_accuracy_list in cbiq_accuracy_lists:
        plt.subplot(1, 5, i)
        plt.plot(timestamp, cbiq_accuracy_list)
        i = i + 1
    axes = plt.gca()
    axes.set(xlabel='cbiq calculation', ylabel='F1')
    axes.grid()
    plt.legend(graph_legend_aligned)#, loc=(0.75,0.5))
    plot_name = "online_filtered_cbiq_graphs"
    plt.savefig(os.path.join(results_path, plot_name), dpi=600)



# Results22:
"""
from learning.thesis_graphs_utils import *
result_path="/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/thesis_prime/online_classification/sampling rate/10000 chunk size/online_filtering/random_filtering/in_and_out_interpolation/30 background flows"
create_f1_vs_online_filtering(result_path,"validation_accuracy","f1 vs filter size")
"""
def create_f1_vs_online_filtering(results_path, txt_filename, plot_name):
    accuracy_list = []
    graph_legend = ["deepcci", "cbiq", "throughput"]
    graph_legend_aligned = []
    for dir_name in os.listdir(results_path):
        res_dir = os.path.join(results_path, dir_name)
        if not os.path.isdir(res_dir) or "old" in res_dir:
            continue
        for graph_type in graph_legend:
            if graph_type in dir_name:
                graph_legend_aligned.append(graph_type)
        result_path = os.path.join(results_path, dir_name, res_dir)
        accuracy_list.append(get_f1_result_for_online_filtering(result_path, txt_filename))
    plt.cla()  # clear the current axes
    plt.clf()  # clear the current figure
    for i in range(len(accuracy_list)):
        accuracy = sorted(accuracy_list[i], key=lambda tup: tup[0])
        x_axis = [x[0] for x in accuracy]
        y_axis = [x[1] for x in accuracy]
        plt.plot(x_axis, y_axis)
    axes = plt.gca()
    axes.set(xlabel='filter [%]', ylabel='F1')
    axes.grid()
    plt.legend(graph_legend_aligned)#, loc=(0.75,0.5))
    plt.savefig(os.path.join(results_path, plot_name), dpi=600)


def get_f1_result_for_online_filtering(results_path, txt_filename):
    accuracy_list = []
    for dir_name in os.listdir(results_path):
        res_dir = os.path.join(results_path, dir_name)
        if not os.path.isdir(res_dir):
            continue
        if "old" in res_dir:
            continue
        x_axis = re.findall(r'\d+', res_dir)
        res_file = os.path.join(res_dir, txt_filename)
        try:
            with open(res_file) as f:
                accuracy = f.readlines()
                accuracy_list.append((int(x_axis[-1]), float(accuracy[-1])))
        except:
            continue
    return accuracy_list

def get_accuracy_result_for_online_filtering(results_path, txt_filename):
    accuracy_list = []
    for dir_name in os.listdir(results_path):
        res_dir = os.path.join(results_path, dir_name)
        if not os.path.isdir(res_dir):
            continue
        if "old" in res_dir:
            continue
        x_axis = re.findall(r'\d+', res_dir)
        res_file = os.path.join(res_dir, txt_filename)
        try:
            with open(res_file) as f:
                accuracy = f.readlines()
                accuracy_list.append((int(x_axis[-1]), float(accuracy[-1])))
        except:
            continue
    return accuracy_list

# Results23:
"""
from learning.thesis_graphs_utils import *
# Path for original online classification:
# result_path="/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/thesis_prime/online_classification/networks comparison/9499 chunk size/background flows/30 background flows/"
result_path="/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/thesis_prime/online vs offline/0 background flows/"
create_online_vs_offline_graph(result_path,"validation_accuracy","f1 for 0 Background Flows and 10 Seconds sessions")
"""
def create_online_vs_offline_graph(results_path, txt_filename, plot_name):
    f1_list = []
    graph_legend = ["CBIQ", "Throughput", "Deepcci"]
    graph_legend_aligned = []
    for dir_name in os.listdir(results_path):
        res_dir = os.path.join(results_path, dir_name)
        if not os.path.isdir(res_dir) or "old" in res_dir in res_dir:
            continue
        f1 = []
        for sub_dir in os.listdir(os.path.join(results_path, dir_name, res_dir)):
            for graph_type in graph_legend:
                if graph_type in sub_dir:
                    graph_legend_aligned.append(graph_type)
            result_path = os.path.join(results_path, dir_name, res_dir, sub_dir)
            f1.append(get_f1_result(result_path, txt_filename))
        f1_list.append(f1)
    plt.cla()  # clear the current axes
    plt.clf()  # clear the current figure
    offline = [x[0][1] for x in f1_list[0]]
    online = [x[0][1] for x in f1_list[1]]

    plt.figure(figsize=(10, 5))
    ind = np.arange(len(offline))
    width = 0.3
    plt.bar(ind, offline, width)
    plt.bar(ind + width, online, width)
    plt.xticks(ind, graph_legend_aligned)

    axes = plt.gca()
    axes.set(xlabel='Parameter', ylabel='F1')
    axes.grid()
    plt.legend(("online", "offline"), loc="best")
    plt.savefig(os.path.join(results_path, plot_name), dpi=600)

if __name__ == '__main__':
    result_path = "/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/thesis_prime/online_classification/sampling rate/10000 chunk size/online_filtering/random_filtering/in_and_out_interpolation/30 background flows"
    create_f1_vs_online_filtering(result_path, "validation_accuracy", "f1 vs filter size")

    """
    result_path = "/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/thesis_prime/online vs offline/0 background flows/"
    create_online_vs_offline_graph(result_path, "validation_accuracy",
                                   "f1 for 0 Background Flows and 10 Seconds sessions")
    """