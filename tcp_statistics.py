import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.legend import DraggableLegend


class TcpStatistics:
    def __init__(self, title="Throughput", plot_file="TheGraph.png"):

        # Clear plt before starting new statistics, otherwise it add up to the previous one
        plt.cla()
        plt.clf()
        '''
        2D Dictionary with amount of transferred data in each cell
        The time column will be the x axis.
        if no data passed for a connection in a specific slot, 0 will be marked
        Time                conn_1      conn_2      conn_3
        ----                ------      ------      ------
        start_time          length      length      length
        start_time+0.1      length      length      length
        start_time+0.2      length      length      length
        
        '''
        self.length_dict_of_dicts = {}

        self.ax = plt.gca()
        self.title = title
        self.plot_file = plot_file

    # Determine a unique key string for the connection
    def get_key(self, src_addr, src_port, dst_addr, dst_port):
        return "%s_%s_%s_%s" % (src_addr, src_port, dst_addr, dst_port)

    # Helper to separated the IP from the port in tecdump parsing
    def parse_ip_port(self, ipp_str):
        search_obj = re.search(r'(\S+)\.(\d+)$', ipp_str)
        if search_obj:
            return search_obj.group(1), search_obj.group(2)
        else:
            return None

    # Parse a single line from TCP dump file, return important values only
    # Example of a single line:
    # 13:53:23.615538 IP tcpip.48286 > 10.0.8.10.5205: Flags [.], ack 1, win 83, options [nop,nop,TS val 3487210525 ecr 3732313601], length 0
    # @return:
    def parse_line(self, line):
        search_obj = re.search(r'(\S+) IP (\S+) > (\S+): Flags.* length (\d+)', line)
        if search_obj == None:
            return

        # Extract the interesting variables
        time_str = search_obj.group(1)
        src_ip_port = search_obj.group(2)
        src_ip, src_port = self.parse_ip_port(src_ip_port)
        dst_ip_port = search_obj.group(3)
        dst_ip, dst_port = self.parse_ip_port(dst_ip_port)
        length = search_obj.group(4)

        # Take only 100th of a second from the time string
        rounded_time_obj = re.search(r'(\S+\.\d)', time_str)  # TODO: There must be a better way to do this
        rounded_time = rounded_time_obj.group(1)

        if all(v is not None for v in [src_ip, src_port, dst_ip, dst_port]):
            # Look for the dictionary element. If it does not exist, create one
            conn_index = self.get_key(src_ip, src_port, dst_ip, dst_port)
            return conn_index, rounded_time, length

    # Parse a tcpdump file, line by line
    def parse_dump_file(self, file_name):

        # Using readlines()
        file1 = open(file_name, 'r')
        lines = file1.readlines()

        count = 0
        # Strips the newline character
        for line in lines:
            conn_index, time_str, length = self.parse_line(line)
            if int(length) == 0:  # ACK only, ignore
                continue

            # If needed, add an entry for the connection (a new column)
            if not conn_index in self.length_dict_of_dicts.keys():
                conn_length_dict = {}
                self.length_dict_of_dicts[conn_index] = conn_length_dict

            # If needed, add a value for the specific time (a new cell)
            if not time_str in self.length_dict_of_dicts[conn_index].keys():
                self.length_dict_of_dicts[conn_index][time_str] = 0

            self.length_dict_of_dicts[conn_index][time_str] += int(length)

    # Convert the 2D dictionary to a plot, using DataFrame
    def create_plot(self):

        # Get rid of all the short connections (not interesting)
        del_list = []
        for conn_id, conn_dict in self.length_dict_of_dicts.items():
            if len(conn_dict.keys()) < 20:
                del_list.append(conn_id)
        for key in del_list:
            del self.length_dict_of_dicts[key]

        # Create a DataFrame out of the dictionaries
        df = pd.DataFrame(self.length_dict_of_dicts)
        df = df.fillna(0.0)

        # Convert throughput from (Bytes /0.1 sec) to Mbps
        df = df.mul(8 / 100000)

        print(df)
        df.plot(kind='line', ax=self.ax, title=self.title)
        plt.legend().set_draggable(True)
        plt.grid(True)
        plt.xlabel('time')
        plt.ylabel('Throughput (Mbps)')
        plt.title = self.title
        plt.savefig(self.plot_file, dpi=200)

# For testing only. the class is called by simulation_implementation.py
if __name__ == '__main__':
    tcp_stat = TcpStatistics()
    filename = "results/4_reno_4_vegas_10000_qsize@6.8.2020@16-50-18/client_vegas_4.txt"
    tcp_stat.parse_dump_file(filename)
    filename = "results/4_reno_4_vegas_10000_qsize@6.8.2020@16-50-18/client_reno_0.txt"
    tcp_stat.parse_dump_file(filename)
    filename = "results/4_reno_4_vegas_10000_qsize@6.8.2020@16-50-18/client_reno_1.txt"
    tcp_stat.parse_dump_file(filename)
    filename = "results/4_reno_4_vegas_10000_qsize@6.8.2020@16-50-18/client_reno_2.txt"
    tcp_stat.parse_dump_file(filename)
    tcp_stat.create_plot()
    plt.show()
