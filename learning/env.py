import os
absolute_path = r'/home/dean/PycharmProjects/cwnd_clgo_classifier/'
cnn_train_and_test_files_directory = r'classification_data/with_data_repetition/queue_size_500'
# cnn_train_and_test_files_directory = r'classification_data/without_data_repetition'
# training_files_path = os.path.join(absolute_path, cnn_train_and_test_files_directory, r'bbr_cubic_reno_sampling_rate_0.001_rtt_0.1sec_with_tsval_train')
training_files_path = os.path.join(absolute_path, cnn_train_and_test_files_directory, r'unfixed_host_bw_srv_bw_with_random_timing_10_tcp_vegas_background_noise')
# training_files_path = os.path.join(absolute_path, cnn_train_and_test_files_directory, r'bbr_cubic_reno_sampling_rate_0.001_rtt_0.1sec_with_tsval_train_fixed_topology_random_timing')
# training_files_path = os.path.join(absolute_path, cnn_train_and_test_files_directory, r'debugging')
testing_files_path = os.path.join(absolute_path, cnn_train_and_test_files_directory, r'bbr_cubic_reno_sampling_rate_0.001_rtt_0.1sec_with_tsval_test')
training_parameters_path = os.path.join(absolute_path, r'cnn_training_parameters/')
testing_results_path = os.path.join(absolute_path, r'data/test_results/')
graphs_path = os.path.join(absolute_path, r'graphs/average_calculation_with_different_session_duration/')

# consts definitions
NUM_OF_CLASSIFICATION_PARAMETERS = 2 # timestemp & CBIQ
NUM_OF_TIME_SAMPLES = 60000
NUM_OF_HIDDEN_LAYERS = 100
CHUNK_SIZE = 1000
DEEPCCI_NUM_OF_TIME_SAMPLES = int(CHUNK_SIZE / 1000)
NUM_OF_CONGESTION_CONTROL_LABELING = 3 # Reno, Cubic, & BBR
NUM_OF_CONV_FILTERS = 50
NUM_OF_EPOCHS = 100
BATCH_SIZE = 32
TRAINING_VALIDATION_RATIO = 0.3
START_AFTER = 6000
END_BEFORE = 0
IS_DEEPCCI = False
IS_BATCH = True
