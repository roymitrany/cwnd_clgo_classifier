import re


class TcQlenStatistics:
    def __init__(self, file_name):
        self.q_len_bytes_dict = {}
        self.q_len_packets_dict = {}
        self.parse_q_len(file_name)


    def parse_q_len(self, file_name):
        # Using readlines()
        file1 = open(file_name, 'r')
        lines = file1.readlines()

        count = 0
        # Strips the newline character
        for line in lines:
            search_obj = re.search(r'(\S+)\s+(\S+)\s+(\S+)', line)
            if search_obj is None:
                continue

            # Extract the interesting variables
            time_str = search_obj.group(1)
            num_of_bytes_str = search_obj.group(2)
            num_of_packets_str = search_obj.group(3)
            if int(num_of_packets_str) == 0:
                continue
            self.q_len_bytes_dict[time_str] = int(num_of_bytes_str)
            self.q_len_packets_dict[time_str] = int(num_of_packets_str)
