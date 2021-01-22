# PyTorch libraries and modules
import torch
from learning.env import *
#from learning.cnn_training import Net
from learning.results_manager import *
import time
from learning.utils import  *

NUM_OF_CLASSIFICATION_PARAMETERS = 2# 3 # 9  # 7
NUM_OF_TIME_SAMPLES = 10000 # 100 # 1200 # 300 # 501 # 601  # 301 # 602
DATAFRAME_BEGINNING = 30000
DATAFRAME_END = 40000

if __name__ == '__main__':
    # normalization_types = ["StatisticalNormalization", "AbsoluteNormalization1", "AbsoluteNormalization2"]
    normalization_types = ["AbsoluteNormalization1"]
    normalization_counter = 0
    classification_difference = []

    # for normalization_type in [StatisticalNormalization(), AbsoluteNormalization1(), AbsoluteNormalization2()]: # 3 different types of normaliztion (pre- processing)
    for normalization_type in [AbsoluteNormalization1()]: # 3 different types of normaliztion (pre- processing)
        """
        model = torch.load(training_parameters_path + normalization_types[normalization_counter] + '_mytraining.pt')  # loading the model and its parameters.
        # model.eval()
        """
        # st = torch.load(training_parameters_path + normalization_types[normalization_counter] + '_mytraining_state_dict.pt')
        model = torch.load(training_parameters_path + normalization_types[normalization_counter] + '_mytraining.pt')  # loading the model and its parameters.
        # model.load_state_dict(st)
        model.eval()
        unused_parameters = ['In Throughput', 'Out Throughput', 'Connection Num of Drops',
                             'Num of Drops', 'Num of Packets', 'Total Bytes in Queue']
        res_mgr = ResultsManager(testing_files_path, normalization_type, 60000, unused_parameters)
        data_labeling = res_mgr.get_train_df()
        dataframe_arr = res_mgr.get_normalized_df_list()

        """
        for csv_file in dataframe_arr: # maybe not necessary
            csv_file = csv_file.drop(csv_file.index[NUM_OF_TIME_SAMPLES:])  # remove samples that were taken after the conventional measuring time.
            csv_file.dropna(inplace=True, how='all')  # remove empty lines after deleting them.
            csv_file = csv_file.fillna((csv_file.shift() + csv_file.shift(-1)) / 2)  # takes care of missing values.
        """

        test_x = [dataframe.to_numpy() for dataframe in dataframe_arr]
        test_x = np.array(test_x)
        test_x = test_x.reshape(len(test_x), 1, NUM_OF_TIME_SAMPLES, NUM_OF_CLASSIFICATION_PARAMETERS)
        test_x = torch.from_numpy(test_x)
        with torch.no_grad():
            output = model(test_x.type('torch.FloatTensor'))
        softmax = torch.exp(output).cpu()
        prob = list(softmax.numpy())
        predictions = np.argmax(prob, axis=1)
        """
        sample_submission = pd.DataFrame().reindex_like(trainning_labeling)
        sample_submission.drop(sample_submission.shape[0]-len(predictions))
        # sample_submission.dropna(inplace=True, how='all')
        """
        testing_labeling = data_labeling.copy()
        testing_labeling['label'] = predictions
        testing_labeling.head()
        data_labeling.to_csv(testing_results_path + '\\' + normalization_types[normalization_counter] + time.strftime("_%d_%m_%Y") + '_training_labeling.csv', index=False)
        testing_labeling.to_csv(testing_results_path + '\\' + normalization_types[normalization_counter] + time.strftime("_%d_%m_%Y") + '_testing_labeling.csv', index=False)

        # comparing "train" and "sample_submission":
        classification_difference.append((testing_labeling[testing_labeling != data_labeling].count()[1]) / len(testing_labeling))
        # print(classification_difference.count()[1] / len(testing_labeling))
        normalization_counter+=1

        train_y = np.array(data_labeling['label'].values)
        train_y = train_y.astype(float)
        train_y = torch.from_numpy(train_y)
        testing_accuracy = accuracy_single_sample(output, train_y, topk=(1,))

    print(classification_difference)
    print(testing_accuracy)