import os
from enum import Enum
from time import sleep
from learning import model_training
from learning.consts import  *

class NetEnum(Enum):
    MY_NET = 1
    DEEPCCI_NET = 2
    FULLY_CONNECTED_NET = 3

class SimParams:
    def __init__(self, data_path, absolute_path, results_path, model_path, diverse_data_path, csv_filename, is_data_sample, is_diverse_data, is_full_session, sleep_duration, save_model_pt):
        self.data_path = data_path
        self.absolute_path = absolute_path
        self.results_path = results_path
        self.model_path = model_path
        self.diverse_data_path = diverse_data_path
        self.csv_filename = csv_filename
        self.is_data_sample = is_data_sample
        self.is_diverse_data = is_diverse_data
        self.is_full_session = is_full_session
        self.sleep_duration = sleep_duration
        self.save_model_pt = save_model_pt


class ModelParams:
    def __init__(self, num_of_congestion_controls, num_of_time_samples, net_type, bg_flow, chunk_size, is_batch):
        # self.net = net_enum.name
        self.num_of_congestion_controls = num_of_congestion_controls
        self.num_of_time_samples = num_of_time_samples
        self.net_type = net_type
        self.bg_flow = bg_flow
        self.chunk_size = chunk_size
        self.is_batch = is_batch


class NetType:
    def __init__(self, net_enum, unused_parameters, deepcci_num_of_time_samples=None):
        if net_enum == NetEnum.MY_NET:
            self.net = "my_net"
        elif net_enum == NetEnum.DEEPCCI_NET:
            self.net = "deepcci_net"
            self.deepcci_num_of_time_samples = deepcci_num_of_time_samples
        else:
            self.net = "fully_connected_net"
        self.parameters, self.unused_parameters_list = unused_parameters
        self.num_of_classification_parameters = NUM_OF_CLASSIFICATION_PARAMETERS - len(self.unused_parameters_list)

    def get_net(self):
        return os.path.join(self.net, self.parameters)

    def get_unused_parameters(self):
        return self.unused_parameters_list

    def get_num_of_classification_parameters(self):
        return self.num_of_classification_parameters

    def get_deepcci_num_of_time_samples(self):
        if self.net == "deepcci_net":
            return deepcci_num_of_time_samples
        else:
            return None

def get_net_types():
    net_types = []
    for net_enum in NetEnum:
        if net_enum == NetEnum.MY_NET:
            #for unused_parameters in [CBIQ_UNUSED_PARAMETERS, IN_THROUGHPUT_UNUSED_PARAMETERS, OUT_THROUGHPUT_UNUSED_PARAMETERS, THROUGHPUT_UNUSED_PARAMETERS, CAPTURE_UNUSED_PARAMETERS, ALL_PARAMETERS_UNUSED_PARAMETERS, DEEPCCI_UNUSED_PARAMETERS]:
            #for unused_parameters in [CBIQ_UNUSED_PARAMETERS]:
            #for unused_parameters in [CBIQ_UNUSED_PARAMETERS, DEEPCCI_UNUSED_PARAMETERS, ALL_PARAMETERS_UNUSED_PARAMETERS, IN_THROUGHPUT_UNUSED_PARAMETERS, OUT_THROUGHPUT_UNUSED_PARAMETERS, CAPTURE_UNUSED_PARAMETERS, THROUGHPUT_UNUSED_PARAMETERS]:
            for unused_parameters in [CBIQ_UNUSED_PARAMETERS, THROUGHPUT_UNUSED_PARAMETERS]:
                net_types.append(NetType(net_enum, unused_parameters))
        elif net_enum == NetEnum.DEEPCCI_NET:
            net_types.append(NetType(net_enum, DEEPCCI_UNUSED_PARAMETERS))
        else:
            continue
    return net_types

def get_results_path(absolute_path, bg_flow, filter, net_type, chunk_size):
    results_path = os.path.join(absolute_path, str(bg_flow) + " background flows", str(filter) + " filter",
                                net_type, str(chunk_size))
    return results_path

if __name__ == '__main__':
    # Automatic graphs generation:
    # sleep(60*60*30)
    num_of_congestion_controls = 3
    # 10 seconds (bottleneck vs no bottleneck graphs):
    num_of_time_samples = 10000 # 60000
    """
    data_paths = [D_10S_3CC_0F_B_PATH, D_10S_3CC_0F_NB_PATH]
    absolute_result_paths = [os.path.join(ABSOLUTE_PATH,
                               r'graphs/thesis_prime/physical/10 seconds/bottleneck vs no bottleneck/bottleneck'),
                            os.path.join(ABSOLUTE_PATH,
                                         r'graphs/thesis_prime/physical/10 seconds/bottleneck vs no bottleneck/no bottleneck')
                            ]
    chunk_sizes = [1000]
    """
    data_paths = [D_10S_3CC_09F_B_PATH_NEW]
    absolute_result_paths = [os.path.join(ABSOLUTE_PATH,
                               r'graphs/thesis_prime/physical/10 seconds/bottleneck vs no bottleneck/1_sub_all')]
    chunk_sizes = [100]
    """
    # 60 seconds (chunk sizes graphs):
    num_of_time_samples = 60000 # 10000
    data_paths = [D_60S_3CC_0F_0BG_B_PATH, D_60S_3CC_0F_0BG_NB_PATH]
    absolute_result_paths = os.path.join(ABSOLUTE_PATH,
                               r'graphs/thesis_prime/physical/60 seconds/bottleneck vs no bottleneck/bottleneck',
                               os.path.join(ABSOLUTE_PATH,
                               r'graphs/thesis_prime/physical/60 seconds/bottleneck vs no bottleneck/no bottleneck')
    chunk_sizes = [1, 10, 100, 500, 1000, 5000, 10000, 30000, 60000]
    """
    model_path = os.path.join(ABSOLUTE_PATH,
                              r'graphs/thesis_prime/classification of different datasets using a single trained model- in multiple routers- Results15/model/state_dict.pt')
    diverse_data_path = [r'filtered_0', r'filtered_0.5', r'filtered_0.9']
    if IS_SAMPLE_RATE:
        csv_filename = "random"
        csv_filename = "ConnStat"
    else:
        csv_filename = "single_connection"
    bg_flows = [0]  # [0, 15, 30, 75]
    filters = [0]
    sleep_duration = 0
    for data_path, absolute_result_path in zip(data_paths, absolute_result_paths):
        for bg_flow in bg_flows:
            for net_type in get_net_types():
                for filter in filters:
                    for chunk_size in chunk_sizes:
                        if chunk_size < 1000:
                            deepcci_num_of_time_samples = 10
                        else:
                            deepcci_num_of_time_samples = int(chunk_size / 1000)
                        results_path = get_results_path(absolute_result_path, bg_flow, filter, net_type.get_net(), chunk_size)
                        sim_params = SimParams(data_path, absolute_result_path, results_path, model_path, diverse_data_path, csv_filename, IS_DATA_SAMPLE, IS_DIVERSE_DATA, IS_FULL_SESSION, sleep_duration, SAVE_MODEL_PT)
                        model_params = ModelParams(num_of_congestion_controls, num_of_time_samples, net_type, bg_flow, chunk_size, IS_BATCH)
                        model_training.main_train_and_validate(sim_params, model_params, DEVICE)
