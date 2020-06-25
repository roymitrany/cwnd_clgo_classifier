import re
import sys

import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import cycler

from tcpdump_statistics import TcpdumpStatistics


class SingleConnStatistics:
    def __init__(self, ingress_file_name, egress_file_name, rtr_q_filename, graph_file_name):
        self.conn_df = self.create_df(ingress_file_name, egress_file_name, rtr_q_filename)
        print(self.conn_df)
        self.create_plots(graph_file_name)

    def create_df(self, ingress_file_name, egress_file_name, rtr_q_filename):
        in_throughput_df = self.parse_dump_file(ingress_file_name)
        out_throughput_df = self.parse_dump_file(egress_file_name)
        connection_df = pd.concat([in_throughput_df, out_throughput_df], axis=1)  # Outer join between in and out df
        connection_df.columns = ['In Throughput', 'in_total', 'Out Throughput', 'out_total']

        # The gap between the total in and the total out indicates what's in the queue
        connection_df['Conn. Bytes in Queue'] = connection_df['in_total'] - connection_df['out_total']

        # Add qdisc columns, only for existing keys (inner join)
        qdisc_df = pd.read_csv(rtr_q_filename, sep="\t", header=None)
        qdisc_df.columns = ['Time', 'Total Bytes in Queue', 'Num of Packets', 'Num of Drops']
        qdisc_df = qdisc_df.set_index('Time')
        connection_df = connection_df.join(qdisc_df, lsuffix='_caller')
        return connection_df

    @staticmethod
    def parse_dump_lines(lines):
        # totals_dict = {}
        time_list = []
        length_list = []

        for line in lines:
            conn_index, time_str, length, ts_val = TcpdumpStatistics.parse_line(line)
            if int(length) == 0:  # ACK only, ignore
                continue

            # Take only 10th of a second from the time string:
            rounded_time_str = time_str[0:-5]
            time_list.append(rounded_time_str)
            length_list.append(float(length))

        df = pd.DataFrame({'Time': time_list, 'Thoughput': length_list})
        df = df.groupby(['Time']).sum()
        df.sort_index()
        df['Total'] = df.cumsum()  # Add Total column that indicates the number of bytes passed so far
        df['Thoughput'] = df['Thoughput'].map(lambda num: num * 8 / 100000)

        return df

    def parse_dump_file(self, file_name):

        # Using readlines()
        file = open(file_name, 'r')
        lines = file.readlines()
        lines = self.reduce_lines(lines)
        lines = self.reduce_retransmissions(lines)
        return self.parse_dump_lines(lines)

    def create_plots(self, graph_file_name):

        fig, (throughput_ax, q_disc_ax) = plt.subplots(2, figsize=(10, 10))
        self.conn_df.plot(kind='line', ax=throughput_ax, y=['In Throughput', 'Out Throughput'],
                          title="Throughput vs. Bytes in Queue")
        throughput_ax.legend(loc=2)
        throughput_ax.set(xlabel='time', ylabel='Throughput (Mbps)')
        throughput_ax.grid()
        ax4 = throughput_ax.twinx()  # instantiate a second axes that shares the same x-axis.
        cm = cycler('color', 'r')
        ax4.set_prop_cycle(cm)
        ax4.set(ylabel='Bytes')
        self.conn_df.plot(kind='line', ax=ax4, y=['Conn. Bytes in Queue'])
        ax4.legend(loc=1)

        self.conn_df.plot(kind='line', ax=q_disc_ax, y=['Num of Drops'], color="red")
        q_disc_ax.set(ylabel='Drops (pkts)')
        q_disc_ax.grid()
        q_disc_ax.legend(loc=2)
        ax5 = q_disc_ax.twinx()  # instantiate a second axes that shares the same x-axis.
        ax5.set(ylabel='Bytes')
        self.conn_df.plot(kind='line', ax=ax5, y=['Conn. Bytes in Queue', 'Total Bytes in Queue'],
                          title="Drops vs. Bytes in Queue")
        ax5.legend(loc=1)
        plt.savefig(graph_file_name)
        plt.show()

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
    def reduce_retransmissions(lines):
        # The method assumes single connection. If lines are from multiple connections, two messages
        # from two different connections with the same seq number will be interpreted as retransmissions
        # If data was sent multiple times, the method keeps only the last retransmission.
        reduced_lines = []
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
            transmission_dict[seq] = line
        return reduced_lines + list(transmission_dict.values())


if __name__ == '__main__':

    if len(sys.argv) == 3:
        test_name = sys.argv[1]
        host_name = sys.argv[2]
        in_file = "results/%s/client_%s.txt" % (test_name, host_name)
        out_file = "results/%s/server_%s.txt" % (test_name, host_name)
        rtr_file = "results/%s/rtr_q.txt" % test_name
        graph_file_name = "results/%s/BIQ_%s.png" % (test_name, host_name)
    else:
        in_file = "test_files/client_reno_0.txt"
        out_file = "test_files/server_reno_0.txt"
        rtr_file = "test_files/rtr_q.txt"
        graph_file_name = "test_files/BIQ.png"
    # in_file = "test_files/in_short.txt"
    # out_file = "test_files/out_short.txt"
    q_line_obj = SingleConnStatistics(in_file, out_file, rtr_file, graph_file_name)
