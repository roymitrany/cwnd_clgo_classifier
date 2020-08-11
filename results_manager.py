import os
import re
import glob
from dataclasses import dataclass
from typing import Dict

import pandas as pd

# Constants
train_file_name = "train.csv"
topo_file_name = "topo_params.txt"
min_num_of_rows = 600


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
            single_exec_df_list = list()
            for csv_file in val.csv_files:
                csv_filename = os.path.join(val.res_path, csv_file)
                orig_conn_stat_df = pd.read_csv(csv_filename, index_col=None, header=0)

                # If the df does not have minimum rows, take it out of the list and continue
                if orig_conn_stat_df['In Throughput'].count() < min_num_of_rows:
                    val.csv_files.remove(csv_file)
                    continue

                fix_conn_stat_df = orig_conn_stat_df.head(min_num_of_rows)
                single_exec_df_list.append(fix_conn_stat_df)
                if "single_connection_stat_bbr" in csv_file:
                    train_list.append(["bbr",0])
                elif "single_connection_stat_cubic" in csv_file:
                    train_list.append(["cubic", 1])
                elif "single_connection_stat_reno" in csv_file:
                    train_list.append(["reno", 2])

            # Normalize all values from the single execution, according to cell location
            # e.g. sum all values in df[261,3], and in every df, divide the current value with the sum
            for row in range(min_num_of_rows):
                for col in single_exec_df_list[0].columns:
                    # sum the cell values
                    sum=0
                    for single_exec_df in single_exec_df_list:
                        sum+=single_exec_df.at[row,col]

                    if sum>0.01:
                        # divide each cell by sum
                        for single_exec_df in single_exec_df_list:
                            single_exec_df.loc[row,col] = single_exec_df.at[row,col]/sum

            for single_exec_df in single_exec_df_list:
                self.connection_stat_df_list.append(single_exec_df)

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
    len_list = list()
    for df in conn_stat_dfl:
        len_list.append(df['In Throughput'].count())
    print("done")