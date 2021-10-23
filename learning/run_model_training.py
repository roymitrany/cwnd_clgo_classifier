from time import sleep
from learning import model_training
from learning.consts import  *
from learning.env import *


class SimParams:
    def __init__(self, data_path, results_path, model_path, diverse_data_path, csv_filename, is_data_sample, is_diverse_data, sleep_duration):
        self.data_path = data_path
        self.results_path = results_path
        self.model_path = model_path
        self.diverse_data_path = diverse_data_path
        self.csv_filename = csv_filename
        self.is_data_sample = is_data_sample
        self.is_diverse_data = is_diverse_data
        self.sleep_duration = sleep_duration


class ModelParams:
    def __init__(self, net_enum, num_of_congestion_controls, num_of_time_samples, unused_parameters, bg_flow, chunk_size, deepcci_num_of_time_samples, num_of_classification_parameters, is_fully_connected, is_batch):
        self.net = net_enum.name
        self.num_of_congestion_controls = num_of_congestion_controls
        self.num_of_time_samples = num_of_time_samples
        self.num_of_classification_parameters = num_of_classification_parameters
        self.unused_parameters = unused_parameters
        self.bg_flow = bg_flow
        self.chunk_size = chunk_size
        self.deepcci_num_of_time_samples = deepcci_num_of_time_samples
        self.is_fully_connected = is_fully_connected
        self.is_batch = is_batch



class NetType:
    def __init__(self, NetEnum):
        self.net = NetEnum.name


if __name__ == '__main__':
    # Automatic graphs generation:
    sleep(60*60*30)
    num_of_congestion_controls = 3
    num_of_time_samples = 60000
    algorithms_list = ["CBIQ", "Deepcci", "Throughput"]
    unused_parameters_list = [CBIQ_UNUSED_PARAMETERS, DEEPCCI_UNUSED_PARAMETERS, THROUGHPUT_UNUSED_PARAMETERS]
    algorithms_dict = dict(zip(algorithms_list, range(len(algorithms_list))))
    algorithms_map = list(zip(algorithms_list, unused_parameters_list))
    data_paths = [D_60S_3CC_0F_0BG_B_PATH]
    result_path = os.path.join(absolute_path,
                               r'graphs/thesis_prime/physical/60 seconds/diverse chunk size/bottleneck')
    diverse_data_path = diverse_training_folder
    if IS_SAMPLE_RATE:
        csv_filename = "random"
    else:
        csv_filename = "single_connection"
    is_data_sample = IS_SAMPLE
    is_diverse_data = IS_DIVERSE
    is_batch = IS_BATCH
    bg_flows = [0]  # [0, 15, 30, 75]
    chunk_sizes = [1, 10, 100, 500, 1000, 5000, 10000, 30000, 60000]
    filters = [0]
    sleep_duration = 0
    is_fully_connected = False
    for data_path in data_paths:
        for bg_flow in bg_flows:
            for (algorithm, unused_parameters) in algorithms_map:
                if algorithm == "Deepcci":
                    is_deepcci = True
                else:
                    is_deepcci = False
                num_of_classification_parameters = NUM_OF_CLASSIFICATION_PARAMETERS - len(unused_parameters)
                for filter in filters:
                    for chunk_size in chunk_sizes:
                        if chunk_size < 1000:
                            deepcci_num_of_time_samples = 10
                        else:
                            deepcci_num_of_time_samples = int(chunk_size / 1000)
                        results_path = os.path.join(result_path, str(bg_flow) + " background flows", str(filter) + " filter", algorithm, str(chunk_size))
                        sim_params = SimParams(data_path, results_path, model_path, diverse_data_path, csv_filename, is_data_sample, is_diverse_data, sleep_duration)
                        model_params = ModelParams(num_of_congestion_controls, num_of_time_samples, unused_parameters, bg_flow, chunk_size, deepcci_num_of_time_samples, num_of_classification_parameters, is_fully_connected, is_batch)
                        model_training.main_train_and_validate(sim_params, model_params, DEVICE)
