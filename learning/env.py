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
cnn_train_and_test_files_directory = r'classification_data/with_data_repetition/queue_size_500/tcp_noise'
# cnn_train_and_test_files_directory = r'classification_data/without_data_repetition'
# training_files_path = os.path.join(absolute_path, cnn_train_and_test_files_directory, r'bbr_cubic_reno_sampling_rate_0.001_rtt_0.1sec_with_tsval_train')
training_files_path = os.path.join(absolute_path, cnn_train_and_test_files_directory, r'START_AFTER/0_bbr_cubic_reno_background_flows')
#training_files_path = os.path.join(absolute_path, cnn_train_and_test_files_directory, r'0_bbr_cubic_reno_background_flows_small_rtt_queue_size_500')
# training_files_path = os.path.join(absolute_path, cnn_train_and_test_files_directory, r'10_vegas_background_flows')
# training_files_path = os.path.join(absolute_path, cnn_train_and_test_files_directory, r'debugging')
testing_files_path = os.path.join(absolute_path, cnn_train_and_test_files_directory, r'bbr_cubic_reno_sampling_rate_0.001_rtt_0.1sec_with_tsval_test')
training_parameters_path = os.path.join(absolute_path, r'cnn_training_parameters/')
testing_results_path = os.path.join(absolute_path, r'data/test_results/')
#graphs_path = os.path.join(absolute_path, r'graphs/unfixed_session_duration/bbr_cubic_reno_background_tcp_flows/15_background_tcp_flows/')
graphs_path = os.path.join(absolute_path, r'graphs/unfixed_session_duration/bbr_cubic_reno_background_tcp_flows_with_all_parameters/per_type/START_AFTER/0_background_tcp_flows_small_rtt/')
#graphs_path = os.path.join(absolute_path, r'graphs/unfixed_session_duration/START_AFTER/bbr_cubic_reno_background_tcp_flows_with_all_parameters/75_background_tcp_flows/')
# graphs_path = os.path.join(absolute_path, r'graphs/unfixed_session_duration/10_background_tcp_vegas_flows/')

# consts definitions
#NUM_OF_CLASSIFICATION_PARAMETERS = 2 # timestemp & CBIQ
NUM_OF_CLASSIFICATION_PARAMETERS = 10 # timestemp & CBIQ
NUM_OF_TIME_SAMPLES = 60000
NUM_OF_HIDDEN_LAYERS = 100
CHUNK_SIZE = 5000
DEEPCCI_NUM_OF_TIME_SAMPLES = int(CHUNK_SIZE / 1000)
NUM_OF_CONGESTION_CONTROL_LABELING = 3 # Reno, Cubic, & BBR
NUM_OF_CONV_FILTERS = 50
NUM_OF_EPOCHS = 100
NUM_OF_BATCHES = 10
BATCH_SIZE = 32
TRAINING_VALIDATION_RATIO = 0.3
START_AFTER = 6000
END_BEFORE = 0
IS_SHUFFLE = False
IS_DEEPCCI = False
IS_BATCH = True

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