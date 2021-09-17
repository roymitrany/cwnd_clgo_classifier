import sys
import subprocess
from subprocess import PIPE, Popen
import os
import torch

"""
global absolute_path
global cnn_train_and_test_files_directory
global training_files_path
global testing_files_path
global training_parameters_path
global testing_results_path
global graphs_path
global IS_DEEPCCI
"""

absolute_path = r'/home/dean/PycharmProjects/cwnd_clgo_classifier/'
#cnn_train_and_test_files_directory = r'classification_data/with_data_repetition/queue_size_500/'
#cnn_train_and_test_files_directory = r'classification_data/with_data_repetition/queue_size_500/tcp_noise/'
cnn_train_and_test_files_directory = r'classification_data/online_classification/'
#training_files_path = os.path.join(absolute_path, cnn_train_and_test_files_directory, r'thesis_new_topology/75_background_flows_new')
#training_files_path = os.path.join(absolute_path, cnn_train_and_test_files_directory, r'75_bbr_cubic_reno_background_flows')
#training_files_path = os.path.join(absolute_path, cnn_train_and_test_files_directory, r'old cbiq calculation/30_background_flows')
#training_files_path = os.path.join(absolute_path, cnn_train_and_test_files_directory, r'15_background_flows')
# training_files_path = '/data_disk/tso_0_75_bg_flows'
#training_files_path = '/data_disk/0.5_filtered_cbiq_interpolate_after_calculation'
#training_files_path = '/data_disk/online filtering/random_interpolation_filter_0.5'
training_files_path = '/data_disk/online filtering/with retransmission/random_interpolation_filter_0'
training_files_path = '/data_disk/classification_data/with_data_repetition/queue_size_500/tcp_noise/START_AFTER/75_bbr_cubic_reno_background_flows'
training_files_path = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/asymetric traffic/filtered_data'
#training_files_path = '/data_disk/physical data/diverse_bg/filtered_data_0_filter'
#training_files_path = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/discrete_bg_high_bw/filtered_data_0.9_filter'
#training_files_path = '/data_disk/debugging'
#training_files_path = os.path.join(absolute_path, cnn_train_and_test_files_directory, r'class_data')
#training_files_path = os.path.join(absolute_path, cnn_train_and_test_files_directory, r'30_bbr_cubic_reno_background_flows')
#diverse_training_folder = [r'75_bbr_cubic_reno_background_flows', r'30_bbr_cubic_reno_background_flows', r'15_bbr_cubic_reno_background_flows', r'0_bbr_cubic_reno_background_flows']
diverse_training_folder = [r'thesis_new_topology/75_background_flows_new', r'tcp_noise/75_bbr_cubic_reno_background_flows']
#training_files_path = os.path.join(absolute_path, cnn_train_and_test_files_directory)
#training_files_path = os.path.join(absolute_path, cnn_train_and_test_files_directory, r'debugging')
testing_files_path = os.path.join(absolute_path, cnn_train_and_test_files_directory, r'bbr_cubic_reno_sampling_rate_0.001_rtt_0.1sec_with_tsval_test')
#training_parameters_path = os.path.join(absolute_path, r'cnn_training_parameters/')
testing_results_path = os.path.join(absolute_path, r'data/test_results/')
#graphs_path = os.path.join(absolute_path,r'graphs/Thesis/test_only_scalability/diverse_training/f1_deepcci/20seconds/different_background_flows/')
#graphs_path = os.path.join(absolute_path,r'graphs/Thesis/test_only_scalability/diverse_training/f1_throughput/20seconds/')
#graphs_path = os.path.join(absolute_path,r'graphs/Thesis/new_topology/all_parameters/20seconds/')
#graphs_path = os.path.join(absolute_path,r'graphs/Thesis/new_topology/multiple_rtr/test_only/20seconds/all_parameters/rtr2_rtr1/')
#graphs_path = os.path.join(absolute_path,r'graphs/Thesis/test_only_scalability/diverse_testing/all_parameters/20seconds/0_background_flows/')
#graphs_path = os.path.join(absolute_path,r'graphs/Thesis/new_topology/multiple_rtr/15_background_flows/20seconds/cbiq/')
#graphs_path = os.path.join(absolute_path,r'graphs/Thesis/new_topology/multiple_rtr/diverse/20seconds/cbiq/')
#graphs_path = os.path.join(absolute_path,r'graphs/thesis_prime/f1 vs number of flows (10 seconds)- Results3/155_background_tcp_flows/')
#graphs_path = os.path.join(absolute_path,r'graphs/thesis_prime/classification of different datasets using a single trained model- in multiple routers- Results15/rtr2/')
#graphs_path = os.path.join(absolute_path,r'graphs/thesis_prime/f1- for each parameter seperately/')
#graphs_path = os.path.join(absolute_path,r'graphs/thesis_prime/online_classification/sampling rate/9499 chunk size/15_background flows/')
# graphs_path = os.path.join(absolute_path,r'graphs/thesis_prime/online_classification/sampling rate/9499 chunk size/debugging')
#graphs_path = os.path.join(absolute_path,r'graphs/thesis_prime/Accuracy VS Session Duration (0 background flows)- Results1/')
#model_path = os.path.join(absolute_path, r'graphs/Thesis/test_only_scalability/f1_all_parameters_model/10seconds_75_background_flows/state_dict.pt')
#model_path = os.path.join(absolute_path, r'graphs/Thesis/test_only_scalability/diverse_training/f1_throughput/20seconds/different_background_flows_model/state_dict.pt')
#model_path = os.path.join(absolute_path, r'graphs/Thesis/test_only_scalability/diverse_testing/all_parameters/20seconds/0_background_flows/model/state_dict.pt')
# model_path = os.path.join(absolute_path, r'graphs/Thesis/new_topology/multiple_rtr/diverse/20seconds/cbiq/model2/state_dict.pt')
model_path = os.path.join(absolute_path, r'graphs/thesis_prime/classification of different datasets using a single trained model- in multiple routers- Results15/model/state_dict.pt')
#model_path = os.path.join(absolute_path, r'graphs/Thesis/new_topology/multiple_rtr/test_only/20seconds/all_parameters/rtr2_rtr1/state_dict.pt')

# consts definitions
NUM_OF_CLASSIFICATION_PARAMETERS = 2 # timestemp & CBIQ
#NUM_OF_CLASSIFICATION_PARAMETERS = 3 # timestemp & throughput
#NUM_OF_CLASSIFICATION_PARAMETERS = 5
NUM_OF_TIME_SAMPLES = 10000 #60000# 9499 # 9499 # 60000
NUM_OF_HIDDEN_LAYERS = 100
CHUNK_SIZE = 10000#9499 # 9499
DEEPCCI_NUM_OF_TIME_SAMPLES = int(CHUNK_SIZE / 1000)
NUM_OF_CONGESTION_CONTROL_LABELING = 3 # Reno, Cubic, & BBR
NUM_OF_CONV_FILTERS = 50
NUM_OF_EPOCHS = 100
NUM_OF_BATCHES = 10
BATCH_SIZE = 32
TRAINING_VALIDATION_RATIO = 0.3
START_AFTER = 6000
END_BEFORE = 0
BG_FLOW = 30 #10 # 10
#graphs_path = os.path.join(absolute_path, r'graphs/thesis_prime/online_classification/sampling rate/10000 chunk size', "online_filtering/random_filtering/in_and_out_interpolation/30 background flows/with retransmission", str(BG_FLOW) + "_bg_flows")
graphs_path = os.path.join(absolute_path, r'graphs/thesis_prime/physical_classification/asymetric traffic', str(BG_FLOW) + "_bg_flows_")
# BG_FLOWS = [0, 1, 5, 10, 20, 25]
IS_SHUFFLE = False
IS_DEEPCCI = False
IS_FULLY_CONNECTED = False
IS_BATCH = True
IS_TEST_ONLY = False
IS_DIVERSE = False
IS_SAMPLE = False
IS_SAMPLE_RATE = True

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
if __name__ == '__main__':
    # Automatic graphs generation:
    for IS_DEEPCCI in [False, True]:
        number_of_flows = [0, 15, 30, 75]
        for flow in number_of_flows:
            dir = str(flow) + '_bbr_cubic_reno_background_flows'
            training_files_path = os.path.join(absolute_path, cnn_train_and_test_files_directory, dir)
            dir = str(flow) + '_background_tcp_flows/'
            graphs_path = os.path.join(absolute_path,
                                       r'graphs/unfixed_session_duration/bbr_cubic_reno_background_tcp_flows_with_all_parameters/' + dir)
            for CHUNK_SIZE in [1000, 3000, 6000, 10000, 30000, 60000]:
                cmd = ['python', 'model_training.py']
                theproc = subprocess.Popen([sys.executable, "model_training.py"]).wait()
"""