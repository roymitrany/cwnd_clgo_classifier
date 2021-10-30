
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
            my_net_all_parameters_accuracy_list.append((int(x_axis[0]), float(accuracy[-1])))
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
                my_net_all_parameters_scatter.append(get_test_only_graph(os.path.join(results_path, dir_name), txt_filename, plot_name, True))
            else:
                my_net_all_parameters_scatter.append(get_test_only_graph(os.path.join(results_path, dir_name), txt_filename, plot_name, False))
        else:
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
        plt.scatter(x_axis, y_axis, color = "red")
    axes = plt.gca()
    axes.set(xlabel='number of flows', ylabel='F1')
    axes.grid()
    plt.legend(graph_legend_aligned)#, loc=(0.75,0.5))
    plt.title(plot_name)
    plt.savefig(os.path.join(results_path, plot_name), dpi=600)


def get_f1_vs_background_flows(results_path, txt_filename, plot_name):
    my_net_all_parameters_accuracy_list = []
    for dir_name in os.listdir(results_path):
        res_dir = os.path.join(results_path, dir_name)
        if not os.path.isdir(res_dir):
            continue
        if "old" in res_dir or "model" in res_dir:
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
    plt.legend(("rtr2", "rtr1"), loc="best")
    plt.savefig(os.path.join(results_path, plot_name), dpi=600)

    """
    from learning.utils import *
    result_path="/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/Thesis/new_topology/multiple_rtr/30_background_flows/"
    create_multiple_rtr_graph(result_path,"validation_accuracy","F1 for 30 Background Flows", "20")
    """

def create_diverse_multiple_rtr_graph(results_path, txt_filename, plot_name, session_duration):
    f1_list = []
    graph_legend = ["5parameters", "cbiq", "deepcci", "throughput"]
    graph_legend_aligned = []
    for dir_name in os.listdir(results_path):
        res_dir = os.path.join(results_path, dir_name)
        if not os.path.isdir(res_dir) or "old" in res_dir or "model" in res_dir:
            continue
        for graph_type in graph_legend:
            if graph_type in dir_name:
                graph_legend_aligned.append(graph_type)
        f1_list.append(get_f1_vs_background_flows(res_dir, txt_filename, plot_name))
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
    plt.legend(("rtr2", "rtr1"), loc="best")
    plt.savefig(os.path.join(results_path, plot_name), dpi=600)

    """
    from learning.utils import *
    result_path="/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/Thesis/new_topology/multiple_rtr/30_background_flows/"
    create_multiple_rtr_graph(result_path,"validation_accuracy","F1 for 30 Background Flows", "20")
    """
