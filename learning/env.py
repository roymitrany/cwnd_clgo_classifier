import os
absolute_path = r'C:\Users\deanc\PycharmProjects\Congestion_Control_classifier_GitHub\\'
cnn_train_and_test_files_directory = r'cnn_train_and_test_files\\'
training_files_path = os.path.join(absolute_path, cnn_train_and_test_files_directory, 'bbr_reno_cubic_bic_vegas_westwood_train_files')
testing_files_path = os.path.join(absolute_path, cnn_train_and_test_files_directory, 'bbr_reno_cubic_bic_vegas_westwood_test_files')
training_parameters_path = os.path.join(absolute_path, 'cnn_training_parameters\\')
testing_results_path = os.path.join(absolute_path, 'test_results')