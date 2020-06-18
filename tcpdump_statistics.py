import re
from typing import Dict


class TcpdumpStatistics:
    def __init__(self, port_algo_dict):

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
        self.ts_val_dict_of_lists = {}
        self.last_ts_val_dict = {}

        self.port_algo_dict: Dict = port_algo_dict

    def get_key(self, src_addr, src_port, dst_addr, dst_port):
        # Check if we can tell the used algo (we should)
        if int(src_port) in self.port_algo_dict.keys():
            algo_str = self.port_algo_dict[int(src_port)] + '_'
        elif int(dst_port) in self.port_algo_dict.keys():
            algo_str = self.port_algo_dict[int(dst_port)] + '_'
        else:
            algo_str = ''

        # Determine a unique key string for the connection:
        return "%s%s_%s_%s_%s" % (algo_str, src_addr, src_port, dst_addr, dst_port)

    def parse_ip_port(self, ipp_str):
        # Auxiliary function to separate the IP from the port in tcpdump parsing:
        search_obj = re.search(r'(\S+)\.(\d+)$', ipp_str)
        if search_obj:
            return search_obj.group(1), search_obj.group(2)
        else:
            return None

    def parse_line(self, line):
        # Parse a single line from TCP dump file, return important values only:
        # Example of a single line:
        # 13:53:23.615538 IP tcpip.48286 > 10.0.8.10.5205: Flags [.], ack 1, win 83, options [nop,nop,TS val 3487210525 ecr 3732313601], length 0
        # search_obj = re.search(r'(\S+) IP (\S+) > (\S+): Flags.* length (\d+)', line)
        search_obj = re.search(r'(\S+) IP (\S+) > (\S+): Flags.*TS val (\d+) .* length (\d+)', line)
        if search_obj is None:
            return '0', '0', '0', '0'

        # Extract the interesting variables
        time_str = search_obj.group(1)
        src_ip_port = search_obj.group(2)
        src_ip, src_port = self.parse_ip_port(src_ip_port)
        dst_ip_port = search_obj.group(3)
        dst_ip, dst_port = self.parse_ip_port(dst_ip_port)
        ts_val = search_obj.group(4)
        length = search_obj.group(5)

        if all(v is not None for v in [src_ip, src_port, dst_ip, dst_port]):
            # Look for the dictionary element. If it does not exist, create one
            conn_index = self.get_key(src_ip, src_port, dst_ip, dst_port)
            return conn_index, time_str, length, ts_val
        else:
            return '0', '0', '0', '0'

    # Parse a tcpdump file, line by line
    def parse_dump_file(self, file_name):

        # Using readlines()
        file1 = open(file_name, 'r')
        lines = file1.readlines()

        count = 0
        # Strips the newline character
        for line in lines:
            conn_index, time_str, length, ts_val = self.parse_line(line)
            if int(length) == 0:  # ACK only, ignore
                continue

            ##### Process packet length
            # for length, take only 10th of a second from the time string
            rounded_time_obj = re.search(r'(\S+\.\d)', time_str)
            rounded_time = rounded_time_obj.group(1)

            # If needed, add an entry for the connection (a new column)
            if not conn_index in self.length_dict_of_dicts.keys():
                conn_length_dict = {}
                self.length_dict_of_dicts[conn_index] = conn_length_dict

            # If needed, add a value for the specific time (a new cell)
            if not rounded_time in self.length_dict_of_dicts[conn_index].keys():
                self.length_dict_of_dicts[conn_index][rounded_time] = 0

            self.length_dict_of_dicts[conn_index][rounded_time] += int(length)

            #### Process TS Val
            if not conn_index in self.ts_val_dict_of_lists.keys():
                ts_val_list = []
                self.ts_val_dict_of_lists[conn_index] = ts_val_list
                self.last_ts_val_dict[conn_index] = float(ts_val)  # Initialize the first element
            self.ts_val_dict_of_lists[conn_index].append(float(ts_val) - self.last_ts_val_dict[conn_index])
            self.last_ts_val_dict[conn_index] = float(ts_val)
