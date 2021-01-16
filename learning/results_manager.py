import abc
import os
import re
import glob
from dataclasses import dataclass
import numpy as np
from typing import Dict

import pandas as pd
from env import *

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

    def __init__(self, results_path, normilizer: Normalizer, min_num_of_rows, dataframe_beginning, dataframe_end):
        """The init function does all the building of the collections, using all results sub folders under
        :param results_path: A String with full path to results location
        """
        self.normalizer = normilizer
        self.res_folder_dict = dict()
        # Create a dictionary that reflects the results file structure
        # create a list of subfolders under results dir
        for dir_name in os.listdir(results_path):
            res_dir = os.path.join(results_path, dir_name)
            if not os.path.isdir(res_dir):
                continue
            csv_files_list = glob.glob(os.path.join(res_dir, "single_connection_stat*"))
            self.res_folder_dict[dir_name] = ResFolder(res_dir, csv_files_list)

        """
        # Added on Saturday:
        # Delete rows containing empty cells and get the number of rows afterwards:
        dataframe_min_rows = min_num_of_rows
        for (iter_name, res_folder) in self.res_folder_dict.items():
            for csv_file in res_folder.csv_files_list:
                csv_filename = os.path.join(res_folder.res_path, csv_file)
                with open(csv_filename) as f:
                    stat_df = pd.read_csv(csv_filename, index_col=None, header=0)
                    nan_value = float("NaN")
                    stat_df.replace("", nan_value, inplace=True)
                    stat_df.dropna(how='any', inplace=True)  # remove empty lines after deleting them.
                    row_count = sum(1 for row in f)
                    if row_count < dataframe_min_rows:
                        dataframe_min_rows = row_count
        self.num_of_rows = dataframe_min_rows
        """

        # Build dataframe array and train array
        train_list = list()
        for (iter_name, res_folder) in self.res_folder_dict.items():
            for csv_file in res_folder.csv_files_list:
                csv_filename = os.path.join(res_folder.res_path, csv_file)
                with open(csv_filename) as f:
                    row_count = sum(1 for row in f)
                    stat_df = pd.read_csv(csv_filename, index_col=None, header=0)
                    #stat_df = stat_df.drop(stat_df.index[min_num_of_rows:])  # remove samples that were taken after the conventional measuring time.
                    stat_df.dropna(inplace=True, how='all')  # remove empty lines after deleting them.
                    """
                    nan_value = float("NaN")
                    stat_df.replace("", nan_value, inplace=True)
                    stat_df.dropna(how='any', inplace=True)  # remove empty lines after deleting them.
                    """
                    stat_df = stat_df.take(stat_df.index[dataframe_beginning:dataframe_end])  # remove samples that were taken after the conventional measuring time.

                for i in range(0, row_count, min_num_of_rows):
                # for i in range(row_count, 0, -min_num_of_rows):
                # if i < row_count /10:
                #     continue
                    """
                    conn_stat_df = pd.read_csv(csv_filename, index_col=None, header=0,
                                               skiprows=range(1, i+1), nrows=min_num_of_rows)
                    """
                    # conn_stat_df = stat_df[min_num_of_rows*i : min_num_of_rows*(i+1)]

                    conn_stat_df = stat_df[i : (i + min_num_of_rows)]
                    #conn_stat_df = stat_df[:dataframe_min_rows]

                    # If the df does not have minimum rows, take it out of the list and continue
                    #if conn_stat_df['In Throughput'].count() < min_num_of_rows:
                    #conn_stat_df = conn_stat_df.fillna((conn_stat_df.shift() + conn_stat_df.shift(-1)) / 2)  # takes care of missing values.

                    if conn_stat_df.shape[0] < min_num_of_rows:
                            continue

                    """
                    index_beginning = len(conn_stat_df) // 2
                    df_indexes = range(index_beginning, index_beginning + min_num_of_rows, 1)
                    conn_stat_df = conn_stat_df.take(df_indexes)
                    """

                    """
                    conn_stat_df = conn_stat_df.drop(
                    columns=['timestamp', 'Out Throughput', 'Connection Num of Drops', 'CBIQ', 'In Throughput',
                             'Num of Drops', 'Num of Packets', 'Total Bytes in Queue'])
                    """

                    conn_stat_df = conn_stat_df.drop(
                        columns=['In Throughput', 'Out Throughput', 'Connection Num of Drops',
                                 'Num of Drops', 'Num of Packets', 'Total Bytes in Queue'])

                    if "single_connection_stat_bbr" in csv_file:
                        train_list.append(["bbr", 0])
                    elif "single_connection_stat_cubic" in csv_file:
                        train_list.append(["cubic", 1])
                    elif "single_connection_stat_reno" in csv_file:
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

                    self.normalizer.add_result(conn_stat_df, iter_name)
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
