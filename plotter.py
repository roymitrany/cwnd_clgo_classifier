from typing import Dict

import pandas as pd
import matplotlib.pyplot as plt

from tc_qlen_statistics import TcQlenStatistics
from tcpdump_statistics import TcpdumpStatistics


class Plotter:
    def __init__(self, plot_file="Graph.png"):

        # Clear plt before starting new statistics, otherwise it add up to the previous one
        plt.cla()
        plt.clf()

        fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 10))
        fig.suptitle('Everything')
        self.throughput_ax = ax1
        self.TSVal_ax = ax2
        plt.grid(True)

        self.plot_file = plot_file

    def create_throughput_plot(self, tcpdump_stat: TcpdumpStatistics, tc_qlen_stat:TcQlenStatistics):
        # Convert the 2D dictionary to a plot, using DataFrame:
        # Get rid of all the short connections (not interesting)
        del_list = []
        for conn_id, conn_dict in tcpdump_stat.length_dict_of_dicts.items():
            if len(conn_dict.keys()) < 10:
                del_list.append(conn_id)
        for key in del_list:
            del tcpdump_stat.length_dict_of_dicts[key]
            del tcpdump_stat.ts_val_dict_of_lists[key]

        ### Plot the throughput
        # Create a DataFrame out of the dictionaries
        df = pd.DataFrame(tcpdump_stat.length_dict_of_dicts)
        df = df.fillna(0.0)

        # Convert throughput from (Bytes /0.1 sec) to Mbps:
        df = df.div(100000 / 8)
        print(df)

        df.plot(kind='line', ax=self.throughput_ax, title='Throughput')

        # Create queue len data frame
        s = pd.Series(tc_qlen_stat.q_len_bytes_dict)
        s.index.name = 'Time'
        print(s)
        q_len_bytes_df = pd.DataFrame(s)
        print(q_len_bytes_df)

        ax3 = self.throughput_ax.twinx()  # instantiate a second axes that shares the same x-axis
        ax3.set(xlabel='time', ylabel='Throughput (Mbps)')
        ax3.legend(loc='upper center', shadow=True, fontsize='xx-small')
        q_len_bytes_df.plot(kind='line', ax=ax3)

    def create_ts_val_plot(self, tcpdump_stat: TcpdumpStatistics):

        # Plot the TS Val
        # Create a list of Series from the TS Val lists
        s_list = []
        for key, l in tcpdump_stat.ts_val_dict_of_lists.items():
            s = pd.Series(l, name=key)
            s_list.append(s)

        # create the data frame
        df = pd.concat(s_list, axis=1)
        # df = df.fillna(0.0)
        df.to_csv("df1.csv")
        self.TSVal_ax.legend(loc='upper center', shadow=True, fontsize='xx-small')
        self.TSVal_ax.set(xlabel='packet num', ylabel='Delta time(ms)')
        df.plot(kind='line', ax=self.TSVal_ax, title="TS Val")

    def save_and_show(self):
        plt.savefig(self.plot_file, dpi=600)


# For testing only: (the class is called by simulation_implementation.py)
if __name__ == '__main__':
    port_algo_dict = {'5201': 'reno'}
    tcp_stat = TcpdumpStatistics(port_algo_dict)
    filename = "test_input.txt"
    tcp_stat.parse_dump_file(filename)
    q_filname = "test_qlen.txt"
    tcp_stat.parse_q_len(q_filname)
    '''filename = "results/4_reno_4_vegas_10000_qsize@6.8.2020@16-50-18/client_reno_0.txt"
    tcp_stat.parse_dump_file(filename)
    filename = "results/4_reno_4_vegas_10000_qsize@6.8.2020@16-50-18/client_reno_1.txt"
    tcp_stat.parse_dump_file(filename)
    filename = "results/4_reno_4_vegas_10000_qsize@6.8.2020@16-50-18/client_reno_2.txt"
    tcp_stat.parse_dump_file(filename)'''
    tcp_stat.create_plot()
    plt.show()
