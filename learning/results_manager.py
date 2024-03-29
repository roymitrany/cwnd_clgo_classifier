import abc
import os
import random
import re
import glob

import numpy
from dataclasses import dataclass
import numpy as np
from typing import Dict

import pandas as pd
from learning.env import *

@dataclass
class ResFolder:
    res_path: str
    csv_files_list: list


class Normalizer(abc.ABC):

    def __init__(self):
        self.normalized_df_list = []

    @abc.abstractmethod
    def add_result(self, res, iter_name):
        pass

    @abc.abstractmethod
    def normalize(self):
        pass


class StatisticalNormalization(Normalizer):

    def __init__(self):
        super().__init__()
        self.result_df_list = []

    def add_result(self, res, iter_name):
        self.result_df_list.append(res)

    def normalize(self):
        # concat_df_list = pd.concat([df for df in self.result_df_list], axis=0)
        # mean_df = concat_df_list.mean()
        # std_df = concat_df_list.std()
        # self.normalized_df_list = [((df - mean_df) / np.sqrt(std_df)).fillna(0) for df in self.result_df_list]
        self.normalized_df_list = [((df - df.mean()) / np.sqrt(df.std())).fillna(0) for df in self.result_df_list]


class AbsoluteNormalization1(Normalizer):

    def __init__(self):
        super().__init__()
        self.result_df_list = []

    def add_result(self, res, iter_name):
        self.result_df_list.append(res)

    def normalize(self):
        self.normalized_df_list = [(df / df.max()).fillna(0) for df in self.result_df_list]


class AbsoluteNormalization2(Normalizer):

    def __init__(self):
        super().__init__()
        self.result_df_dict = {}

    def add_result(self, res, iter_name):
        if iter_name not in self.result_df_dict:
            self.result_df_dict[iter_name] = []

        self.result_df_dict[iter_name].append(res)

    def normalize(self):

        # Loop on each simulation iteration, and normalize results according to the iteration
        for iter_name, single_exec_df_list in self.result_df_dict.items():

            # Normalize all values from the single execution, according to cell location
            # e.g. sum all values in df[261,3], and in every df, divide the current value with the sum
            # Create sum df
            sum_df = single_exec_df_list[0]
            for df in single_exec_df_list[1:]:
                sum_df = sum_df + df

            for df in single_exec_df_list:
                normalized_df = df / sum_df
                self.normalized_df_list.append(normalized_df.fillna(0))

            print("Finished normalizing %s" % iter_name)


class ResultsManager:

    def __init__(self, results_path, normilizer: Normalizer, min_num_of_rows, unused_parameters, chunk_size, is_diverse, diverse_training_folder = [], start_after = 0, end_before = 0): #, dataframe_beginning=0, dataframe_end=60000):
        """The init function does all the building of the collections, using all results sub folders under
        :param results_path: A String with full path to results location
        """
        self.normalizer = normilizer
        self.res_folder_dict = dict()
        # Create a dictionary that reflects the results file structure
        # create a list of subfolders under results dir
        if is_diverse:
            for dir_name in diverse_training_folder:
                sub_folder = os.path.join(os.path.join(absolute_path, cnn_train_and_test_files_directory), dir_name)
                for sub_dir_name in os.listdir(sub_folder):
                    res_dir = os.path.join(results_path, sub_folder, sub_dir_name)
                    if not os.path.isdir(res_dir):
                        continue
                    csv_files_list = glob.glob(os.path.join(res_dir, "single_connection_stat*"))
                    self.res_folder_dict[sub_dir_name] = ResFolder(res_dir, csv_files_list)
                """
                #for i in range(50):
                    random_subfolder = random.choice(sub_folder)
                    res_dir = os.path.join(results_path, dir_name, random_subfolder)
                    if not os.path.isdir(res_dir):
                        continue
                    csv_files_list = glob.glob(os.path.join(res_dir, "single_connection_stat*"))
                    self.res_folder_dict[random_subfolder] = ResFolder(res_dir, csv_files_list)
                """
        else:
            if IS_SAMPLE_RATE == True:
                for dir_name in os.listdir(results_path):
                    res_dir = os.path.join(results_path, dir_name)
                    if not os.path.isdir(res_dir):
                        continue
                    #csv_files_list = glob.glob(os.path.join(res_dir, "milli*"))
                    csv_files_list = glob.glob(os.path.join(res_dir, "random*"))
                    self.res_folder_dict[dir_name] = ResFolder(res_dir, csv_files_list)
            else:
                for dir_name in os.listdir(results_path):
                    res_dir = os.path.join(results_path, dir_name)
                    if not os.path.isdir(res_dir):
                        continue
                    csv_files_list = glob.glob(os.path.join(res_dir, "single_connection_stat*"))
                    self.res_folder_dict[dir_name] = ResFolder(res_dir, csv_files_list)

        # Build dataframe array and train array
        train_list = list()
        iteration = 0
        if IS_SAMPLE:
            keys = random.sample(range(len(self.res_folder_dict)),250)
        else:
            keys = range(len(self.res_folder_dict))
        for (iter_name, res_folder) in self.res_folder_dict.items():
            iteration += 1
            if not(iteration in keys):
                continue
            if chunk_size < 5000 and iteration > 4: # added for 1 second session duration- overfitting debugging.
                break
            for csv_file in res_folder.csv_files_list:
                csv_filename = os.path.join(res_folder.res_path, csv_file)
                if not ("bbr" in csv_filename or "reno" in csv_filename or "cubic" in csv_filename):
                    continue
                with open(csv_filename) as f:
                    row_count = sum(1 for row in f)
                    stat_df = pd.read_csv(csv_filename, index_col=None, header=0)
                    if row_count - 1 < min_num_of_rows:
                        continue
                    # remove samples taken before all flows have started sessions:
                    #stat_df = stat_df[START_AFTER:]
                    # remove samples that were taken after the conventional measuring time:
                    stat_df = stat_df.take(stat_df.index[row_count - min_num_of_rows - 1:])
                    # keep only samples taken between the random beginning and end of all flows:
                    #stat_df = stat_df[start_after:min_num_of_rows-end_before]

                    if unused_parameters is not None:
                        stat_df = stat_df.drop(columns=unused_parameters)
                    # split dataframe to chunks:
                    # number_of_chunks = stat_df.shape[0] / chunk_size + stat_df.shape[0] % chunk_size
                    number_of_chunks = stat_df.shape[0] / chunk_size
                    for stat_df_chunk in np.array_split(stat_df, number_of_chunks):
                        if not IS_DEEPCCI and NUM_OF_CLASSIFICATION_PARAMETERS != 3:
                            try:
                                # Taking care of CBIQ calculation irregulars:
                                # stat_df_chunk['CBIQ']=stat_df_chunk.where(stat_df_chunk < 1, 0)
                                stat_df_chunk['CBIQ'] = abs(stat_df_chunk['CBIQ'])
                                # stat_df_chunk['CBIQ'] = 0
                                # stat_df_chunk.loc[stat_df_chunk.CBIQ < 1, 'CBIQ'] = 0
                                # stat_df_chunk['CBIQ']=stat_df_chunk['CBIQ'].where(stat_df_chunk['CBIQ'] < 1, 0)
                                stat_df_chunk.loc[stat_df_chunk.CBIQ < 1, 'CBIQ'] = 0
                                #stat_df_chunk.loc[stat_df_chunk.CBIQ == 2896.0, 'CBIQ'] = 0
                                if "new_topology" in csv_filename:
                                    stat_df_chunk['CBIQ'] = 0
                            except:
                                pass

                        if "stat_bbr" in csv_file:
                            train_list.append(["bbr", 0])
                        elif "stat_cubic" in csv_file:
                            train_list.append(["cubic", 1])
                        elif "stat_reno" in csv_file:
                            train_list.append(["reno", 2])
                            """
                        elif "single_connection_stat_vegas" in csv_file:
                            train_list.append(["vegas", 3])
                        elif "single_connection_stat_bic" in csv_file:
                            train_list.append(["bic", 4])
                        elif "single_connection_stat_westwood" in csv_file:
                            train_list.append(["westwood", 5])
                            """
                        else:
                            continue

                        self.normalizer.add_result(stat_df_chunk, iter_name)
                        print("added %s to list" % iter_name)
        self.train_df = pd.DataFrame(train_list, columns=["id", "label"])

        self.normalizer.normalize()

    def get_train_df(self):
        return self.train_df

    def get_normalized_df_list(self):
        return self.normalizer.normalized_df_list

    #def get_num_of_rows(self):
     #   return self.num_of_rows


# For testing only
if __name__ == '__main__':
    normaliz = AbsoluteNormalization2()
    res_mgr = ResultsManager(testing_results_path, normaliz, 30)
    norm_dfl = res_mgr.get_normalized_df_list()
    len_list = list()
    for df in norm_dfl:
        len_list.append(df['In Throughput'].count())
    print("done")
