import os
import json
import re
from enum import Enum
from signal import SIGINT

from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import setLogLevel
from mininet.net import Mininet
from datetime import datetime
from mininet.node import OVSController
from mininet.util import pmonitor

from mytopo import RenoVSVegasTopo


class TcpAlgo(Enum):
    reno = 1
    cubic = 2
    vegas = 3
    westwood = 4
    bbr = 5


class Iperf3Tester:
    def __init__(self, topo, test_name, seconds=10):
        """

        :param topo: The topology class to be used
        :param test_name: the results folder will contain the test name
        :param seconds: The test duration
        """
        self.topo = topo
        self.net = Mininet(topo, controller=OVSController,
                           link=TCLink,
                           autoSetMacs=True)
        self.seconds = seconds
        self.test_name = test_name
        tn = datetime.now()
        time_str = str(tn.hour) + "_" + str(tn.minute) + "_" + str(tn.second)

        # str(ts).split(".")[0][-4:]
        self.res_dirname = os.path.join(os.getcwd(), "results", test_name + "_" + time_str)
        os.mkdir(self.res_dirname, 0o777);
        self.iperf_out_filename = os.path.join(self.res_dirname, "iperf_output.txt")
        self.rtr_q_filename = os.path.join(self.res_dirname, "rtr_q.txt")

        # Initialize resuls dictionary. Keys are the list of clients in the topology
        '''
        client dictionary stores the output strings per client. Output dows not necessarily contain results
        for n clients running m intervals, the dict will look like this:
        client_1: [output_1, output_2 ... output_m]
        client_2: [output_1, output_2 ... output_m]
        .
        .
        .
        client_n: [output_1, output_2 ... output_m]        
        '''
        self.res_dict = {}
        for client in topo.host_list:
            self.res_dict[client] = []

    def change_tcp_cwnd_algo(self, host_idx, tcp_algo):
        """
        :param host_idx:
        :type int:
        :param tcp_algo: a member of TcpAlgo enum e.g. TcpAlgo.westwood
        :type int:
        :return:
        """

        host = self.topo.host_list[host_idx]
        cmd = 'echo %s > /proc/sys/net/ipv4/tcp_congestion_control' % tcp_algo.name
        self.net.getNodeByName(host).cmd(cmd)

    def start_test(self):
        self.net.start()
        CLI(self.net)
        # Modify TCP algo (because iperf3 does not support vegas in -C parameter)
        for host_number in range(0, self.topo.num_of_reno_hosts):
            self.change_tcp_cwnd_algo(host_number, TcpAlgo.reno)
        for host_number in range(self.topo.num_of_reno_hosts,
                                 self.topo.num_of_reno_hosts + self.topo.num_of_vegas_hosts):
            self.change_tcp_cwnd_algo(host_number, TcpAlgo.vegas)

        srv = self.net.getNodeByName(self.topo.srv)
        srv_ip = srv.IP()
        popens = {}
        test_count = 0
        srv_procs = []
        tshark_procs = []
        for client in self.topo.host_list:
            test_port = (5201 + test_count)
            srv_cmd = 'iperf3 -s -p %d&' % test_port
            srv_procs.append(srv.popen(srv_cmd))

            # run tshark in client side, save it into pcap file
            pcap_file_name = os.path.join(self.res_dirname, "host_%s.pcap" % client)
            cmd = 'tshark -i host_int -f "port %d" -w %s' % (test_port, pcap_file_name)
            tshark_procs.append((self.net.getNodeByName(client)).popen(cmd))
            # Run iperf3
            cmd = 'iperf3 -c %s -t %d -p %d' % (srv_ip, self.seconds, 5201 + test_count)
            popens[client] = (self.net.getNodeByName(client)).popen(cmd)
            test_count += 1
        rtr = self.net.getNodeByName(self.topo.rtr)
        q_proc = rtr.popen('python queue_len_poller.py r-srv %s' % self.rtr_q_filename)
        # We need to wait util all commands are completed
        for client, line in pmonitor(popens, timeoutms=1000):
            if client:
                print('<%s>: %s' % (client, line), )
            if client in self.res_dict:
                self.res_dict[client].append(line)
        # Kill the server's iperf -s processes, the router's queue monitor anf tsharks
        for process in tshark_procs:
            process.send_signal(SIGINT)
        for process in srv_procs:
            process.send_signal(SIGINT)
        q_proc.send_signal(SIGINT)

        self.net.stop()
        print(self.res_dict)

        # Save in file as JSON format
        with open(self.iperf_out_filename, 'w') as outfile:
            json_str = json.dumps(self.res_dict)
            outfile.write(json_str)

        outfile.close()


if __name__ == '__main__':
    # Tell mininet to print useful information
    num_of_reno = num_of_vegas = 1
    setLogLevel('info')
    rv_topo = RenoVSVegasTopo(num_of_reno, num_of_vegas, srv_bw=209, srv_delay=150000, rtr_queue_size=951)
    tester = Iperf3Tester(rv_topo, "%s_%s" % (str(num_of_reno), str(num_of_reno)), 10)
    tester.start_test()
