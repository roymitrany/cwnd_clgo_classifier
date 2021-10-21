from time import sleep

from learning import model_training
from learning.consts import  *
from learning.env import *
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
    bg_flows = [0]  # [0, 15, 30, 75]
    chunk_sizes = [1, 10, 100, 500, 1000, 5000, 10000, 30000, 60000]
    filters = [0]
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
                        model_training.main(data_path, unused_parameters, bg_flow, IS_SAMPLE_RATE, IS_SAMPLE, is_deepcci, IS_FULLY_CONNECTED, num_of_classification_parameters,
                                            chunk_size, num_of_congestion_controls, num_of_time_samples, DEVICE, results_path, IS_TEST_ONLY, model_path, diverse_training_folder, IS_DIVERSE, deepcci_num_of_time_samples)
