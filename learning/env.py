import os
absolute_path = r'/home/dean/PycharmProjects/cwnd_clgo_classifier/'
cnn_train_and_test_files_directory = r'classification_data/'
training_files_path = os.path.join(absolute_path, cnn_train_and_test_files_directory, r'bbr_cubic_reno_sampling_rate_0.001_rtt_0.1sec_with_tsval_train')
# training_files_path = os.path.join(absolute_path, cnn_train_and_test_files_directory, r'bbr_cubic_reno_sampling_rate_0.001_rtt_0.1sec_with_tsval_train_fixed_topology_random_timing')
# training_files_path = os.path.join(absolute_path, cnn_train_and_test_files_directory, r'debugging')
testing_files_path = os.path.join(absolute_path, cnn_train_and_test_files_directory, r'bbr_cubic_reno_sampling_rate_0.001_rtt_0.1sec_with_tsval_test')
training_parameters_path = os.path.join(absolute_path, r'cnn_training_parameters/')
testing_results_path = os.path.join(absolute_path, r'data/test_results/')
graphs_path = os.path.join(absolute_path, r'graphs/average_calculation/')
