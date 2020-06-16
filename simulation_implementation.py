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

from simulation_topology import SimulationTopology
from tcp_statistics import TcpStatistics


class TcpAlgorithm(Enum):
    reno = 1
    cubic = 2
    vegas = 3
    westwood = 4
    bbr = 5


class Iperf3Simulator:
    def __init__(self, simulation_topology, simulation_name, seconds=10):
        """
        :param simulation_topology: The topology class to be used
        :param simulation_name: the results folder will contain the test name
        :param seconds: The test duration
        """
        self.file_captures = []
        self.simulation_topology = simulation_topology
        self.net = Mininet(simulation_topology, controller=OVSController, link=TCLink, autoSetMacs=True)
        self.seconds = seconds
        self.simulation_name = simulation_name
        tn = datetime.now()
        time_str = str(tn.month) + "." + str(tn.day) + "." + str(tn.year) + "@" + str(tn.hour) + "-" + str(
            tn.minute) + "-" + str(tn.second)

        # Create results directory, with name includes num of reno num of vegas and time
        self.res_dirname = os.path.join(os.getcwd(), "results", self.simulation_name + "@" + time_str)
        os.mkdir(self.res_dirname, 0o777);

        # Set results file names
        self.iperf_out_filename = os.path.join(self.res_dirname, "iperf_output.txt")
        self.rtr_q_filename = os.path.join(self.res_dirname, "rtr_q.txt")

    def SetCongestionControlAlgorithm(self, host_idx, tcp_algo):
        """
        :param tcp_algo: a member of TcpAlgorithm enum e.g. TcpAlgorithm.westwood
        """

        host = self.simulation_topology.host_list[host_idx]
        cmd = 'echo %s > /proc/sys/net/ipv4/tcp_congestion_control' % tcp_algo.name
        self.net.getNodeByName(host).cmd(cmd)

    def StartSimulation(self):
        self.net.start()
        CLI(self.net)
        # Modify TCP algorithms (because iperf3 does not support vegas in -C parameter):
        for host_number in range(0, self.simulation_topology.num_of_reno_hosts):
            self.SetCongestionControlAlgorithm(host_number, TcpAlgorithm.reno)
        for host_number in range(self.simulation_topology.num_of_reno_hosts,
                                 self.simulation_topology.num_of_reno_hosts + self.simulation_topology.num_of_vegas_hosts):
            self.SetCongestionControlAlgorithm(host_number, TcpAlgorithm.vegas)

        srv = self.net.getNodeByName(self.simulation_topology.srv)
        srv_ip = srv.IP()
        popens = {}
        srv_procs = []
        rtr = self.net.getNodeByName(self.simulation_topology.rtr)

        # Auxiliary loop- including iperf for the servers and initializing monotoring functions using "tcpdump":
        client_counter = 0
        for client in self.simulation_topology.host_list:
            test_port = (5201 + client_counter)
            # In iperf3, each test should have its own server. We have to terminate them at the end,
            # otherwise they are stuck in the system, so we keep the proc nums in a list.
            srv_cmd = 'iperf3 -s -p %d&' % test_port
            srv_procs.append(srv.popen(srv_cmd))

            """
            # Throughput measuring- using tshark:
            # Running tshark on client side, saving to pcap file (a separate pcap file for each client):
            pcap_file_name = os.path.join(self.res_dirname, "host_%s.pcap" % client)
            cmd = 'tshark -i host_int -f "port %d" -w %s -F libpcap&' % (test_port, pcap_file_name)
            (self.net.getNodeByName(client)).cmd(cmd)
            # Running tshark on server side, saving to pcap file (a separate pcap file for each client):
            pcap_file_name = os.path.join(self.res_dirname, "server_%s.pcap" % client)
            cmd = 'tshark -i srv-r -f "port %d" -w %s&' % (test_port, pcap_file_name)
            srv.cmd(cmd)
            """

            # Throughput measuring- using tcpdump:
            # Running tcpdump on client side, saving to txt file (a separate txt file for each client):
            capture_filename = os.path.join(self.res_dirname, "client_%s.txt" % client)
            interface_name = "r-%s" % client
            cmd = "tcpdump -i %s 'tcp port %d'>%s&" % (interface_name, test_port, capture_filename)
            rtr.cmd(cmd)
            # Running tcpdump on server side, saving to txt file (a separate txt file for each client):
            capture_filename = os.path.join(self.res_dirname, "server_%s.txt" % client)
            self.file_captures.append(capture_filename)
            cmd = "tcpdump -i r-srv 'tcp port %d'>%s&" % (test_port, capture_filename)
            rtr.cmd(cmd)
            client_counter += 1

        sleep(10)
        # Traffic generation loop:
        client_counter = 0
        for client in self.simulation_topology.host_list:
            cmd = 'iperf3 -c %s -t %d -p %d' % (srv_ip, self.seconds, 5201 + client_counter)
            popens[client] = (self.net.getNodeByName(client)).popen(cmd)
            client_counter += 1

        # Gather statistics from the router:
        q_proc = rtr.popen('python queue_statistics.py r-srv %s' % self.rtr_q_filename)
        pcap_filename = os.path.join(self.res_dirname, "rtr_srv.pcap")
        rtr.cmd('tshark -i r-srv -w %s -F libpcap&' % pcap_filename)

        # Wait until all commands are completed:
        for client, line in pmonitor(popens, timeoutms=1000):
            if client:
                print('<%s>: %s' % (client, line), )
        # Kill the server's iperf -s processes, the router's queue monitor and tcpdumps:
        sleep(10)
        for process in srv_procs:
            process.send_signal(SIGINT)
        q_proc.send_signal(SIGINT)

        self.net.stop()


if __name__ == '__main__':
    # Tell mininet to print useful information
    num_of_reno = num_of_vegas = 4
    queue_size = 1000
    setLogLevel('info')
    for queue_size in [300]:
        for num_of_reno in range(4,5):
            num_of_vegas = 8-num_of_reno
            # bw is in Mbps, delay in msec, queue size in packets
            simulation_topology = SimulationTopology(num_of_reno, num_of_vegas, host_bw=100, srv_bw=200, srv_delay=15000,
                                                     rtr_queue_size=queue_size)
            simulation_name = "%d_reno_%d_vegas_%d_qsize" % (num_of_reno, num_of_vegas, queue_size)
            simulator = Iperf3Simulator(simulation_topology, simulation_name, 10)
            simulator.StartSimulation()
            tcp_stat = TcpStatistics(
                title='Throughput - %d Reno, %d vegas, %d queue len' % (num_of_reno, num_of_vegas, queue_size),
                plot_file=os.path.join(simulator.res_dirname, 'Throughput.png'),
                plot_file2 = os.path.join(simulator.res_dirname, 'tsval.png'))
            for filename in simulator.file_captures:
                tcp_stat.parse_dump_file(filename)
            tcp_stat.create_plot()

