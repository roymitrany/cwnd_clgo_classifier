import abc
import os
import random
import glob
from dataclasses import dataclass
import numpy as np
import pandas as pd

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
        self.normalized_df_list = [((df - df.mean()) / np.sqrt(df.std())).fillna(0) for df in self.result_df_list]


class AbsoluteNormalization1(Normalizer):
    def __init__(self):
        super().__init__()
        self.result_df_list = []

    def add_result(self, res, iter_name):
        self.result_df_list.append(res)

    def normalize(self, is_deepcci, num_of_classification_parameters):
        if not is_deepcci and num_of_classification_parameters == 2:
            self.normalized_df_list = self.result_df_list
        else:
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
    def __init__(self, sim_params, model_params, normilizer: Normalizer, is_deepcci):
        """The init function does all the building of the collections, using all results sub folders under
        :param results_path: A String with full path to results location
        """
        self.net_type = model_params.net_type
        self.is_deepcci = is_deepcci
        self.normalizer = normilizer
        self.res_folder_dict = dict()
        # Create a dictionary that reflects the results file structure
        # create a list of subfolders under results dir
        self.get_results(sim_params, model_params)
        # Build dataframe array and train array
        self.train_list = list()
        iteration = 0
        if sim_params.is_data_sample:
            keys = random.sample(range(len(self.res_folder_dict)), 50)
        else:
            keys = range(len(self.res_folder_dict))
        for (iter_name, res_folder) in self.res_folder_dict.items():
            iteration += 1
            if not(iteration in keys):
                continue
            for csv_file in res_folder.csv_files_list:
                csv_filename = os.path.join(res_folder.res_path, csv_file)
                if not ("bbr" in csv_filename or "reno" in csv_filename or "cubic" in csv_filename):
                    continue
                with open(csv_filename) as f:
                    row_count = sum(1 for row in f)
                    stat_df = pd.read_csv(csv_filename, index_col=None, header=0)
                    if row_count - 1 < model_params.num_of_time_samples:
                        continue
                    # remove samples that were taken after the conventional measuring time:
                    stat_df = stat_df.take(stat_df.index[row_count - model_params.num_of_time_samples - 1:])
                    self.unused_parameters = self.net_type.get_unused_parameters()
                    if self.unused_parameters is not None:
                        stat_df = stat_df.drop(columns=self.unused_parameters)
                    # split dataframe to chunks:
                    # number_of_chunks = stat_df.shape[0] / chunk_size + stat_df.shape[0] % chunk_size
                    number_of_chunks = stat_df.shape[0] / model_params.chunk_size
                    if not sim_params.is_full_session:
                        stat_df_chunk = random.sample(np.array_split(stat_df, number_of_chunks), 1)[0]
                        self.classify_chunk(csv_file, stat_df_chunk, iter_name)
                    else:
                        for stat_df_chunk in np.array_split(stat_df, number_of_chunks):
                            self.classify_chunk(csv_file, stat_df_chunk, iter_name)

        self.train_df = pd.DataFrame(self.train_list, columns=["id", "label"])
        self.normalizer.normalize(self.is_deepcci, self.net_type.get_num_of_classification_parameters())

    def classify_chunk(self, csv_file, stat_df_chunk, iter_name):
        if "stat_bbr" in csv_file:
            self.train_list.append(["bbr", 0])
        elif "stat_cubic" in csv_file:
            self.train_list.append(["cubic", 1])
        elif "stat_reno" in csv_file:
            self.train_list.append(["reno", 2])
            """
        elif "single_connection_stat_vegas" in csv_file:
            train_list.append(["vegas", 3])
        elif "single_connection_stat_bic" in csv_file:
            train_list.append(["bic", 4])
        elif "single_connection_stat_westwood" in csv_file:
            train_list.append(["westwood", 5])
            """
        else:
            return
        self.normalizer.add_result(stat_df_chunk, iter_name)
        print("added %s to list" % iter_name)

    def get_results(self, sim_params, model_params):
        if sim_params.is_diverse_data:
            for dir_name in sim_params.diverse_data_path:
                sub_folder = os.path.join(os.path.join(sim_params.absolute_path, sim_params.diverse_data_path), dir_name)
                for sub_dir_name in os.listdir(sub_folder):
                    res_dir = os.path.join(sim_params.results_path, sub_folder, sub_dir_name)
                    if not os.path.isdir(res_dir):
                        continue
                    csv_files_list = glob.glob(os.path.join(res_dir, sim_params.csv_filename + "*"))
                    self.res_folder_dict[sub_dir_name] = ResFolder(res_dir, csv_files_list)
        else:
            for dir_name in os.listdir(sim_params.data_path):
                bg = "NumBG_" + str(model_params.bg_flow)
                if bg not in dir_name:
                    continue
                res_dir = os.path.join(sim_params.data_path, dir_name)
                if not os.path.isdir(res_dir):
                    continue
                csv_files_list = glob.glob(os.path.join(res_dir, sim_params.csv_filename + "*"))
                self.res_folder_dict[dir_name] = ResFolder(res_dir, csv_files_list)

    def get_train_df(self):
        return self.train_df

    def get_normalized_df_list(self):
        return self.normalizer.normalized_df_list

# For testing only
if __name__ == '__main__':
    normaliz = AbsoluteNormalization2()
    res_mgr = ResultsManager(testing_results_path, normaliz, 30)
    norm_dfl = res_mgr.get_normalized_df_list()
    len_list = list()
    for df in norm_dfl:
        len_list.append(df['In Throughput'].count())
    print("done")
