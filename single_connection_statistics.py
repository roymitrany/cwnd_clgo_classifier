import re
import sys

import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import cycler

from tcpdump_statistics import TcpdumpStatistics


'''def fix_time(rounded_time_str):
    z = datetime.strptime(rounded_time_str, "%H:%M:%S.%f")
    time_in_tenth = int(10 * (z.hour * 3600 + z.minute * 60 + z.second + z.microsecond / 1000000))
    date_time = datetime.fromtimestamp(time_in_tenth / 10)
    time = date_time.strftime("%H:%M:%S.%f")[0:-5]
    return time'''


class SingleConnStatistics:
    def __init__(self, ingress_file_name, egress_file_name, rtr_q_filename, graph_file_name):
        self.conn_df = self.create_dfs(ingress_file_name, egress_file_name, rtr_q_filename)
        print(self.conn_df)
        self.create_plots(graph_file_name)

    def create_dfs(self, ingress_file_name, egress_file_name, rtr_q_filename):
        in_dict = self.parse_dump_file(ingress_file_name)
        out_dict = self.parse_dump_file(egress_file_name)
        in_ser = pd.Series(in_dict)
        out_ser = pd.Series(out_dict)
        totals_df = pd.DataFrame({'in_total': in_ser, 'out_total': out_ser})

        throughput_df = totals_df.diff()
        throughput_df.columns = ['In Throughput', 'Out Throughput']
        throughput_df = throughput_df.div(100000 / 8)
        totals_df = totals_df.join(throughput_df)

        totals_df['Conn. Bytes in Queue'] = totals_df['in_total'] - totals_df['out_total']

        # Add qdisc columns, only for existinf keys
        qdisc_df = pd.read_csv(rtr_q_filename, sep="\t", header=None)
        qdisc_df.columns = ['Time', 'Total Bytes in Queue', 'Num of Packets', 'Num of Drops']
        #qdisc_df['Time'] = qdisc_df['Time'].map(lambda time: fix_time(time))
        qdisc_df = qdisc_df.set_index('Time')
        totals_df = totals_df.join(qdisc_df, lsuffix='_caller')
        return totals_df

    @staticmethod
    def parse_dump_lines(lines):
        totals_dict = {}

        last_time_in_tenth = -1
        total_bytes = 0
        # Strips the newline character
        count = 0
        for line in lines:
            count += 1
            if count % 1000 == 0:
                print('&', end='')
                if count % 30000 == 0:
                    print(count)
            conn_index, time_str, length, ts_val = TcpdumpStatistics.parse_line(line)
            if int(length) == 0:  # ACK only, ignore
                continue
            total_bytes += int(length)
            # Take only 10th of a second from the time string:
            rounded_time_str = time_str[0:-5]

            totals_dict[rounded_time_str] = total_bytes

        return totals_dict

    def parse_dump_file(self, file_name):

        # Using readlines()
        file = open(file_name, 'r')
        lines = file.readlines()
        lines = self.reduce_lines(lines)
        lines = self.reduce_retransmissions(lines)
        return self.parse_dump_lines(lines)

    def create_plots(self, graph_file_name):

        print("=======CP1=============")
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
        print("=======CP2=============")

        self.conn_df.plot(kind='line', ax=q_disc_ax, y=['Num of Drops'], color="red")
        q_disc_ax.set(ylabel='Drops (pkts)')
        q_disc_ax.grid()
        q_disc_ax.legend(loc=2)
        ax5 = q_disc_ax.twinx()  # instantiate a second axes that shares the same x-axis.
        ax5.set(ylabel='Bytes')
        self.conn_df.plot(kind='line', ax=ax5, y=['Conn. Bytes in Queue', 'Total Bytes in Queue'],
                          title="Drops vs. Bytes in Queue")
        ax5.legend(loc=1)
        print("=======CP3=============")
        plt.savefig(graph_file_name)
        plt.show()

    @staticmethod
    def reduce_lines(lines):
        # Take out all lines with length 0
        count = 0
        conn_count = {}
        for line in lines:
            count += 1
            if count % 1000 == 0:
                print('-', end='')
                if count % 30000 == 0:
                    print(count)
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

        count = 0
        reduced_lines = []
        # loop on file again and remove all the lines that are not interesting
        for line in lines:
            count += 1
            if count % 1000 == 0:
                print('+', end='')
                if count % 30000 == 0:
                    print(count)
            conn_index, time_str, length, ts_val = TcpdumpStatistics.parse_line(line, {})
            if conn_index == our_conn_index:
                reduced_lines.append(line)

        return reduced_lines

        # 09:17:58.297429 IP 10.0.1.10.44848 > 10.0.10.10.5202: Flags [.], seq 1486:2934, ack 1, win 83, options [nop,nop,TS val 4277329349 ecr 645803186], length 1448

    def reduce_retransmissions(self, lines):
        # The method assumes single connection. If lines are from multiple connections, thwo messages
        # from two different connections with the same seq number will be interpreted as retransmissions
        reduced_lines = []
        transmission_dict = {}
        count = 0
        for line in lines:
            count += 1
            if count % 1000 == 0:
                print('^', end='')
                if count % 30000 == 0:
                    print(count)
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
        in_file = "test_files/in_file_test.txt"
        out_file = "test_files/out_file_test.txt"
        rtr_file = "test_files/rtr_q.txt"
        graph_file_name = "test_files/BIQ.png"
    # in_file = "test_files/in_short.txt"
    # out_file = "test_files/out_short.txt"
    q_line_obj = SingleConnStatistics(in_file, out_file, rtr_file, graph_file_name)
