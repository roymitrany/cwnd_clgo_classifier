import re
import sys

import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import cycler, gridspec

from simulation.tcpdump_statistics import TcpdumpStatistics


def get_sec(time_str):
    """Get Seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)


class SingleConnStatistics:
    def __init__(self, ingress_file_name, egress_file_name, rtr_q_filename, graph_file_name, plot_title,
                 generate_graphs, interval_accuracy):
        self.conn_df = self.rolling_df = None
        self.interval_accuracy = interval_accuracy

        self.build_df(ingress_file_name, egress_file_name, rtr_q_filename)
        # print(self.conn_df)
        # self.create_plots(graph_file_name)
        if generate_graphs:
            self.create_plots(graph_file_name, plot_title)

    def build_df(self, ingress_file_name, egress_file_name, rtr_q_filename):

        # Parse inbound file. Extract dropped packets and passed packets for the connection
        file = open(ingress_file_name, 'r')
        lines = file.readlines()
        in_conn_lines = self.reduce_lines(lines)  # take out all the lines that are not related to the connection
        in_passed_lines, in_dropped_lines = self.reduce_dropped_packets(in_conn_lines)

        # Parse outbound file. Extract passed packets for the connection
        file = open(egress_file_name, 'r')
        lines = file.readlines()
        out_conn_lines = self.reduce_lines(lines)  # take out all the lines that are not related to the connection

        # Create several DataFrames from the lines
        dropped_df = self.create_dropped_df(in_dropped_lines, self.interval_accuracy)
        in_throughput_df = self.create_throughput_df(in_conn_lines, self.interval_accuracy)
        out_throughput_df = self.create_throughput_df(out_conn_lines, self.interval_accuracy)
        in_goodput_df = self.create_throughput_df(
            in_passed_lines, self.interval_accuracy)  # Throughput for packets that did not drop
        ts_val_df = self.create_ts_val_df(in_conn_lines, self.interval_accuracy)

        # Consolidate all the DataFrames into one DataFrame
        self.conn_df = pd.concat([in_throughput_df, out_throughput_df, dropped_df, ts_val_df],
                                 axis=1)  # Outer join between in and out df
        self.conn_df.columns = ['In Throughput', 'Out Throughput', 'Connection Num of Drops', 'Send Time Gap']
        self.conn_df.index.name = 'Time'
        self.conn_df.sort_index()
        # Create the Byte in Queue column
        # Add Total column that indicates the number of bytes passed so far
        in_goodput_df['In Total'] = in_goodput_df.cumsum()
        out_throughput_df['Out Total'] = out_throughput_df.cumsum()
        df = pd.concat([in_goodput_df['In Total'], out_throughput_df['Out Total']], axis=1)
        # The gap between the total in and the total out indicates what's in the queue. We want to convert form
        # Mbps to Bytes
        df['CBIQ'] = df['In Total'] - df['Out Total']
        df['CBIQ'] = df['CBIQ'].map(lambda num: num / 8 * 100000)
        self.conn_df = self.conn_df.join(df['CBIQ'], lsuffix='_caller')

        values = {'Connection Num of Drops': 0}
        self.conn_df = self.conn_df.fillna(value=values)

        # Add qdisc columns, only for existing keys (inner join)
        qdisc_df = pd.read_csv(rtr_q_filename, sep="\t", header=None)
        qdisc_df.columns = ['Time', 'Total Bytes in Queue', 'Num of Packets', 'Num of Drops']
        qdisc_df = qdisc_df.set_index('Time')
        self.conn_df = self.conn_df.join(qdisc_df, lsuffix='_caller')

        # Convert the time string into time offset float
        base_timestamp = get_sec(self.conn_df.index[0])
        self.conn_df['timestamp'] = self.conn_df.index.map(mapper=(lambda x: get_sec(x) - base_timestamp))
        self.conn_df = self.conn_df.set_index('timestamp')
        return

    @staticmethod
    def create_ts_val_df(lines, interval_accuracy):
        # TS val indicates the timestamp in which the packet was sent.
        # since we want to maintain the data based on time intervals, we should expect some intervals to include
        # a lot of packets, and others with only few of them.
        # It is hard to process such information, so we will
        # extract the maximal time gap between two sent packets for each interval.
        ts_val_dict = {}
        last_ts_val = 0
        for line in lines:
            conn_index, time_str, length, ts_val = TcpdumpStatistics.parse_line(line)
            ts_val = int(ts_val)
            s_str = '(\S+\.\d{%d})' % interval_accuracy
            rounded_time_obj = re.search(r'%s' % s_str, time_str)
            rounded_time = rounded_time_obj.group(1)
            if last_ts_val == 0:
                last_ts_val = ts_val

            if rounded_time not in ts_val_dict:
                ts_val_dict[rounded_time] = ts_val - last_ts_val
            else:
                ts_val_dict[rounded_time] = max(ts_val_dict[rounded_time], ts_val - last_ts_val)
            last_ts_val = ts_val
        df = pd.DataFrame.from_dict(ts_val_dict, orient='index')
        return df

    @staticmethod
    def create_throughput_df(lines, interval_accuracy):
        time_list = []
        length_list = []

        for line in lines:
            conn_index, time_str, length, ts_val = TcpdumpStatistics.parse_line(line)
            if int(length) == 0:  # ACK only, ignore
                continue

            # Take only 10th of a second from the time string:
            rounded_time_str = time_str[0:0-interval_accuracy]
            time_list.append(rounded_time_str)
            length_list.append(float(length))

        df = pd.DataFrame({'Time': time_list, 'Thoughput': length_list})
        df = df.groupby(['Time']).sum()
        df['Thoughput'] = df['Thoughput'].map(lambda num: num * 8 / 100000)

        return df

    @staticmethod
    def create_dropped_df(dropped_lines, interval_accuracy):
        time_list = []
        for line in dropped_lines:
            conn_index, time_str, length, ts_val = TcpdumpStatistics.parse_line(line)

            # Take only 10th of a second from the time string:
            rounded_time_str = time_str[0:0-interval_accuracy]
            time_list.append(rounded_time_str)

        df = pd.DataFrame({'Time': time_list})
        df = df.groupby(['Time']).size()
        df.columns = ['Num of Drops']
        return df

    @staticmethod
    def reduce_lines(lines):
        # Take out all lines with length 0
        conn_count = {}
        for line in lines:
            # Ignore if length is 0
            if int(line.find('length 0')) > 0:  # ACK only, ignore
                continue
            conn_index, time_str, length, ts_val = TcpdumpStatistics.parse_line(line, {})
            if conn_index in conn_count.keys():
                conn_count[conn_index] += 1
            else:
                conn_count[conn_index] = 1

        # extract the connection index
        our_conn_index = max(conn_count, key=conn_count.get)

        reduced_lines = []
        # loop on file again and add only the interesting lines to the list
        for line in lines:
            conn_index, time_str, length, ts_val = TcpdumpStatistics.parse_line(line, {})
            if conn_index == our_conn_index:
                reduced_lines.append(line)

        return reduced_lines

        # 09:17:58.297429 IP 10.0.1.10.44848 > 10.0.10.10.5202: Flags [.], seq 1486:2934, ack 1, win 83, options [nop,nop,TS val 4277329349 ecr 645803186], length 1448

    @staticmethod
    def reduce_dropped_packets(lines):
        # The method assumes single connection. If lines are from multiple connections, two messages
        # from two different connections with the same seq number will be interpreted as retransmissions
        # If data was sent multiple times, the method keeps only the last retransmission.
        reduced_lines = []
        dropped_lines = []
        transmission_dict = {}
        for line in lines:
            search_obj = re.search(r'.*seq (\d+).* length (\d+)', line)
            # All lines with no seq or no data should be automatically not reduced
            if search_obj is None:
                reduced_lines.append(line)
                continue
            length = search_obj.group(2)
            if int(length) == 0:
                reduced_lines.append(line)
                continue

            seq = int(search_obj.group(1))
            # Map the line to the sequence. If older sequence was there, it will be automatically reduced.
            if seq in transmission_dict:
                dropped_lines.append(transmission_dict[seq])
            transmission_dict[seq] = line
        return reduced_lines + list(transmission_dict.values()), dropped_lines

    def create_plots(self, graph_file_name, plot_title):

        fig2 = plt.figure(constrained_layout=True, figsize=(10, 10))
        fig2.suptitle(plot_title)
        spec2 = gridspec.GridSpec(ncols=1, nrows=3, figure=fig2)
        throughput_ax = fig2.add_subplot(spec2[0, 0])
        q_disc_ax = fig2.add_subplot(spec2[1, 0])
        ts_ax = fig2.add_subplot(spec2[2, 0])
        # max_ax = fig2.add_subplot(spec2[1, 1])
        # f2_ax3 = fig2.add_subplot(spec2[1, 0])
        # f2_ax4 = fig2.add_subplot(spec2[1, 1])

        # fig, (throughput_ax, q_disc_ax) = plt.subplots(2, figsize=(10, 10))

        self.conn_df.plot(kind='line', ax=throughput_ax, y=['In Throughput', 'Out Throughput'],
                          title="Throughput vs. Bytes in Queue")
        throughput_ax.legend(loc=2)
        throughput_ax.set(xlabel='time', ylabel='Throughput (Mbps)')
        throughput_ax.grid()
        ax4 = throughput_ax.twinx()  # instantiate a second axes that shares the same x-axis.
        cm = cycler('color', 'r')
        ax4.set_prop_cycle(cm)
        ax4.set(xlabel='time', ylabel='Bytes')
        self.conn_df.plot(kind='line', ax=ax4, y=['CBIQ'])
        ax4.legend(loc=1)

        self.conn_df.plot(kind='line', ax=q_disc_ax, y=['Connection Num of Drops'], color="red")
        q_disc_ax.set(xlabel='time', ylabel='Drops (pkts)')
        q_disc_ax.grid()
        q_disc_ax.legend(loc=2)
        ax5 = q_disc_ax.twinx()  # instantiate a second axes that shares the same x-axis.
        ax5.set(ylabel='Bytes')
        self.conn_df.plot(kind='line', ax=ax5, y=['CBIQ'],
                          title="Drops vs. Bytes in Queue")
        ax5.legend(loc=1)

        # self.max_cbiq_series.plot(kind='line', ax=max_ax, c='g')
        # self.rolling_df.plot(kind='line', ax=max_ax, c='g', y=['CBIQ'])

        self.conn_df.plot(kind='line', ax=ts_ax, y=['Connection Num of Drops'], color="red")
        ts_ax.set(xlabel='time', ylabel='Drops (pkts)')
        ts_ax.grid()
        ts_ax.legend(loc=2)
        ax6 = ts_ax.twinx()  # instantiate a second axes that shares the same x-axis.
        ax6.set(ylabel='msec')
        self.conn_df.plot(kind='line', ax=ax6, y=['Send Time Gap'],
                          title="Drops vs. Send Time Gap")
        ax6.legend(loc=1)

        plt.savefig(graph_file_name)
        plt.show(block=False)


if __name__ == '__main__':

    if len(sys.argv) == 3:
        test_name = sys.argv[1]
        host_name = sys.argv[2]
        in_file = "results/%s/client_%s.txt" % (test_name, host_name)
        out_file = "results/%s/server_%s.txt" % (test_name, host_name)
        rtr_file = "results/%s/rtr_q.txt" % test_name
        graph_file_name = "results/%s/BIQ_%s.png" % (test_name, host_name)
        q_line_obj = SingleConnStatistics(in_file, out_file, rtr_file, graph_file_name)
        q_line_obj.conn_df.to_csv("results/%s/single_connection_stat_%s.csv" % (test_name, host_name))
    else:
        in_file = "../test_files/in_file_test.txt"
        out_file = "../test_files/out_file_test.txt"
        rtr_file = "../test_files/rtr_q.txt"
        graph_file_name = "test_files/BIQ.png"
        q_line_obj = SingleConnStatistics(in_file, out_file, rtr_file, graph_file_name)
        q_line_obj.conn_df.to_csv("test_files/single_connection_stat.csv")
    # in_file = "test_files/in_short.txt"
    # out_file = "test_files/out_short.txt"
