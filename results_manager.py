import os
import re
import glob
from dataclasses import dataclass
from typing import Dict

import pandas as pd

# Constants
train_file_name = "train.csv"
topo_file_name = "topo_params.txt"


@dataclass
class ResFolder:
    res_path: str
    csv_files: list


class ResultsManager:

    def __init__(self, results_path):
        """The init function does all the building of the collections, using all results sub folders under
        :param results_path: A String with full path to results location
        """
        self.res_folder_dict = dict()
        # Create a dictionary that reflects the results file structure
        # create a list of subfolders under results dir
        for dir_name in os.listdir(results_path):
            res_dir = os.path.join(results_path, dir_name)
            if not os.path.isdir(res_dir):
                continue
            files = os.listdir(res_dir)
            if (train_file_name in files) and (topo_file_name in files):
                # It's a legit results subfolder
                csv_files_list = glob.glob(os.path.join(res_dir, "single_connection_stat*"))
                self.res_folder_dict[dir_name] = ResFolder(res_dir, csv_files_list)

        # Build dataframe array and train array
        self.connection_stat_df_list = list()
        train_list = list()
        for (key, val) in self.res_folder_dict.items():
            csv_files_list = val.csv_files
            for csv_file in csv_files_list:
                csv_filename = os.path.join(val.res_path, csv_file)
                df = pd.read_csv(csv_filename, index_col=None, header=0)
                self.connection_stat_df_list.append(df)
                if "single_connection_stat_bbr" in csv_file:
                    train_list.append(["bbr",0])
                elif "single_connection_stat_cubic" in csv_file:
                    train_list.append(["cubic", 1])
                elif "single_connection_stat_reno" in csv_file:
                    train_list.append(["reno", 2])
        self.train_df = pd.DataFrame(train_list, columns=["id", "label"])

    def get_train_df(self):
        return self.train_df

    def get_connection_stat_df_list(self):
        return self.connection_stat_df_list



# For testing only
if __name__ == '__main__':
    res_mgr = ResultsManager("C:\\Users\\roym\\PycharmProjects\\cwnd_clgo_classifier\\test_results")
    train_df = res_mgr.get_train_df()
    conn_stat_dfl = res_mgr.get_connection_stat_df_list()
    print("done")