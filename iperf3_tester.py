import os
import json
import re
from enum import Enum
from signal import SIGINT
from time import sleep

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
        time_str = str(tn.month) + "_" + str(tn.day) + "_" + str(tn.hour) + "_" + str(tn.minute)

        # Create results directory, with name includes num of reno num of vegas and time
        self.res_dirname = os.path.join(os.getcwd(), "results", test_name + "_" + time_str)
        # self.res_dirname = os.path.join("/tmp/test1")
        # self.res_dirname = os.path.join("/tmp/test1")
        os.mkdir(self.res_dirname, 0o777)

        # Set results file names
        self.iperf_out_filename = os.path.join(self.res_dirname, "iperf_output.txt")
        self.rtr_q_filename = os.path.join(self.res_dirname, "rtr_q.txt")

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
        srv_procs = []
        rtr = self.net.getNodeByName(self.topo.rtr)

        # Preparation loop, including iperf servers and monitoring
        test_count = 0
        for client in self.topo.host_list:
            test_port = (5201 + test_count)
            # In iperf3, each test should have its own server. We have to terminate them t the end,
            # otherwise they are stuck in the system, so we keep the proc nums in a list
            srv_cmd = 'iperf3 -s -p %d&' % test_port
            srv_procs.append(srv.popen(srv_cmd))

            # run tshark in client  side, save it into pcap file. Is a separate pcap per client
            pcap_file_name = os.path.join(self.res_dirname, "client_%s.pcap" % client)
            cmd = 'tshark -i host_int -f "port %d" -w %s&' % (test_port, pcap_file_name)
            (self.net.getNodeByName(client)).cmd(cmd)

            # run tshark in server  side, save it into pcap file. Is a separate pcap per client
            pcap_file_name = os.path.join(self.res_dirname, "server_%s.pcap" % client)
            cmd = 'tshark -i srv-r -f "port %d" -w %s&' % (test_port, pcap_file_name)
            srv.cmd(cmd)
            test_count += 1

        sleep(10)
        # Packet generation loop
        test_count = 0
        for client in self.topo.host_list:
            # Run iperf3
            cmd = 'iperf3 -c %s -t %d -p %d' % (srv_ip, self.seconds, 5201 + test_count)
            popens[client] = (self.net.getNodeByName(client)).popen(cmd)
            test_count += 1

        # Gather statistics in the router
        q_proc = rtr.popen('python queue_len_poller.py r-srv %s' % self.rtr_q_filename)
        # pcap_file_name = os.path.join(self.res_dirname, "rtr_srv.pcap")
        # rtr.cmd('tshark -i r-srv -w %s -F libpcap&' % pcap_file_name)

        # We need to wait util all commands are completed
        for client, line in pmonitor(popens, timeoutms=1000):
            if client:
                print('<%s>: %s' % (client, line), )
        # Kill the server's iperf -s processes, the router's queue monitor anf tsharks
        CLI(self.net)
        sleep(10)
        for process in srv_procs:
            process.send_signal(SIGINT)
        #for process in tshark_procs:
        #    process.send_signal(SIGINT)
        #    process.send_signal(SIGINT)
        q_proc.send_signal(SIGINT)

        self.net.stop()


if __name__ == '__main__':
    # Tell mininet to print useful information
    num_of_reno = num_of_vegas = 4
    setLogLevel('info')
    # bw is in Mbps, delay in msec, queue size in packets
    rv_topo = RenoVSVegasTopo(num_of_reno, num_of_vegas, host_bw = 100, srv_bw=209, srv_delay=0, rtr_queue_size=100)
    tester = Iperf3Tester(rv_topo, "%s_%s" % (str(num_of_reno), str(num_of_reno)), 10)
    tester.start_test()
