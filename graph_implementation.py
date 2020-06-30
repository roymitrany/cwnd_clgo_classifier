from dictionaries import Dict
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from cycler import cycler

from tc_qdisc_statistics import TcQdiscStatistics
from tcpdump_statistics import TcpdumpStatistics


class GraphImplementation:
    def __init__(self, tcpdump_statistsics, tc_qdisc_statistics, plot_file_name="Graph.png", plot_fig_name="Congestion Control Statistics"):

        # Clear plt before starting new statistics, otherwise it add up to the previous one:
        # plt.cla() # Clear the current axes.
        # plt.clf() # Clear the current figure.

        fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 10))
        # fig.suptitle('Congestion Control Algorithms Statistics')
        fig.suptitle(plot_fig_name)
        ax1.set(xlabel='time', ylabel='Throughput (Mbps)')
        ax2.set(xlabel='time', ylabel='Number of drops')
        # ax3.set(xlabel='num of packets', ylabel='Delta Time (ms)')
        self.throughput_ax = ax1
        self.drop_ax = ax2
        # self.TSVal_ax = ax3

        self.plot_file_name = plot_file_name

        self.create_throughput_plot(tcpdump_statistsics, tc_qdisc_statistics)
        self.create_drop_plot(tc_qdisc_statistics)
        # graph_implementation.create_ts_val_plot(tcpdump_statistsics)
        self.throughput_ax.grid()
        self.drop_ax.grid()
        self.save_and_show()

    def create_throughput_plot(self, tcpdump_statistsics, tc_qdisc_statistics):
        # Convert the 2D dictionary to a plot, using DataFrame:
        # Get rid of all the short connections (not interesting)
        del_list = []
        for conn_id, conn_dict in tcpdump_statistsics.throughput_dict_of_dicts.items():
            if len(conn_dict.keys()) < 10:
                del_list.append(conn_id)
        for key in del_list:
            del tcpdump_statistsics.throughput_dict_of_dicts[key]
            del tcpdump_statistsics.ts_val_dict_of_lists[key]

        ### Plot the throughput:
        # Create a DataFrame out of the dictionaries:
        throughput_df = pd.DataFrame(tcpdump_statistsics.throughput_dict_of_dicts)
        # throughput_df = throughput_df.fillna(0.0)

        # Convert throughput from (Bytes /0.1 sec) to Mbps:
        throughput_df = throughput_df.div(100000 / 8)
        throughput_df.plot(kind='line', ax=self.throughput_ax, title='Throughput')

        # Create queue length data frame:
        queue_length_series = pd.Series(tc_qdisc_statistics.q_len_packets_dict)
        queue_length_series.index.name = 'Time'
        queue_length_df = pd.DataFrame(queue_length_series)
        queue_length_df.columns = ['Port QLen']

        ax3 = self.throughput_ax.twinx()  # instantiate a second axes that shares the same x-axis.
        ax3.set(xlabel='time', ylabel='Number of packets')
        ax3.legend(loc='upper center', shadow=True, fontsize='xx-small')
        # cm = plt.get_cmap('gist_rainbow')
        # ax3.set_prop_cycle('color',plt.cm.Spectral(np.linspace(0,1,30)))
        cm = cycler('color', 'r')
        ax3.set_prop_cycle(cm)
        queue_length_df.plot(kind='line', ax=ax3)

    def create_drop_plot(self, tc_qdisc_statistics):

        # Create drops data frame:
        drops_series = pd.Series(tc_qdisc_statistics.q_drops_dict)
        drops_series.index.name = 'Time'
        drops_df = pd.DataFrame(drops_series)
        drops_df.columns = ['Port Drops']
        drops_df.plot(kind='line', ax=self.drop_ax, title='Drops')

        # Create queue length data frame:
        queue_length_series = pd.Series(tc_qdisc_statistics.q_len_packets_dict)
        queue_length_series.index.name = 'Time'
        queue_length_df = pd.DataFrame(queue_length_series)
        queue_length_df.columns = ['Port QLen']

        ax4 = self.drop_ax.twinx()  # instantiate a second axes that shares the same x-axis.
        ax4.set(xlabel='time', ylabel='Number of packets')
        # ax4.legend(loc='upper center', shadow=True, fontsize='xx-small')
        cm = cycler('color', 'r')
        ax4.set_prop_cycle(cm)
        queue_length_df.plot(kind='line', ax=ax4)

    def create_ts_val_plot(self, tcpdump_statistsics):
        ### Plot the TS Val:
        # Create a list of Series from the TS Val lists:
        s_list = []
        for key, l in tcpdump_statistsics.ts_val_dict_of_lists.items():
            queue_length_series = pd.Series(l, name=key)
            s_list.append(queue_length_series)

        # Create the data frame:
        throughput_df = pd.concat(s_list, axis=1)
        # throughput_dataframe = throughput_dataframe.fillna(0.0)
        throughput_df.to_csv("df1.csv")
        self.TSVal_ax.legend(loc='upper center', shadow=True, fontsize='xx-small')
        self.TSVal_ax.set(xlabel='packet num', ylabel='Delta time(ms)')
        throughput_df.plot(kind='line', ax=self.TSVal_ax, title="TS Val")

    def save_and_show(self):
        plt.savefig(self.plot_file_name, dpi=600)
        plt.show()


# For testing only: (the class is called by simulation_implementation.py)
if __name__ == '__main__':
    port_algo_dict = {'5201': 'reno'}
    tcp_stat = TcpdumpStatistics(port_algo_dict)
    filename = "test_files/test_input.txt"
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
    # plt.show()
