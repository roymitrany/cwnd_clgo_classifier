import os
import json
import re
from typing import Dict

import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
from cycler import cycler

from plotter import Plotter
from simulation_topology import SimulationTopology
from tc_qlen_statistics import TcQlenStatistics
from tcpdump_statistics import TcpdumpStatistics


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

        # Create results directory, with name includes num of clients for each algo, and time:
        self.res_dirname = os.path.join(os.getcwd(), "results", self.simulation_name + "@" + time_str)
        os.mkdir(self.res_dirname, 0o777)

        # Set results file names:
        self.iperf_out_filename = os.path.join(self.res_dirname, "iperf_output.txt")
        self.rtr_q_filename = os.path.join(self.res_dirname, "rtr_q.txt")

        self.port_algo_dict = {}


    def SetCongestionControlAlgorithm(self, host, tcp_algo):
        """
        :param tcp_algo: a member of TcpAlgorithm enum e.g. TcpAlgorithm.westwood
        """
        cmd = 'echo %s > /proc/sys/net/ipv4/tcp_congestion_control' % tcp_algo
        self.net.getNodeByName(host).cmd(cmd)

    def StartSimulation(self):
        self.net.start()
        CLI(self.net)

        srv = self.net.getNodeByName(self.simulation_topology.srv)
        srv_ip = srv.IP()
        popens = {}
        srv_procs = []
        rtr = self.net.getNodeByName(self.simulation_topology.rtr)

        # Auxiliary loop- including iperf for the servers and initializing monotoring functions using "tcpdump":
        client_counter = 0
        for client in self.simulation_topology.host_list:
            # Modify TCP algorithms (because iperf3 does not support vegas in -C parameter):
            cwnd_algo = client[0:client.find("_")]
            self.SetCongestionControlAlgorithm(client, cwnd_algo)

            test_port = (5201 + client_counter)

            # Map test port to algo. This will serve us in results processing
            self.port_algo_dict[test_port] = cwnd_algo

            # In iperf3, each test should have its own server. We have to terminate them at the end,
            # otherwise they are stuck in the system, so we keep the proc nums in a list.
            srv_cmd = 'iperf3 -s -p %d &' % test_port
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

        sleep(5)
        # Traffic generation loop:
        client_counter = 0
        for client in self.simulation_topology.host_list:
            cwnd_algo = client[0:client.find("_")]
            # cmd = 'iperf3 -c %s -t %d -p %d -C %s' % (srv_ip, self.seconds, 5201 + client_counter, self.congestion_control_algorithm[client_counter % 2])
            cmd = 'iperf3 -c %s -t %d -p %d -C %s' % (srv_ip, self.seconds, 5201 + client_counter,cwnd_algo)
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
        sleep(5)
        for process in srv_procs:
            process.send_signal(SIGINT)
        q_proc.send_signal(SIGINT)
        # CLI(self.net)
        self.net.stop()

    """
    def GenerateIntervalQueue(self, start, step, count):
        return [start + i * step for i in xrange(count)]
    
    def PrintGraphs(self):
        with open(self.rtr_q_filename, 'r') as f:
            lines = f.readlines()
            queue_size = [float(line.split()[0]) for line in lines]
            # Measuring samples interval = 0.01.
            interval = self.GenerateIntervalQueue(0, 0.01, len(queue_size))
            f.close()
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('interval [s]')
        ax1.set_ylabel('throughput [bytes/s]')
        
        # Print graphs for each client (using its own txt file):
        # SEPERATION_SPREAD_SIZE = 2
        # NUM_COLORS = SEPERATION_SPREAD_SIZE * (NUMBER_OF_RENO_HOSTS + NUMBER_OF_VEGAS_HOSTS)
        # cm = plt.get_cmap('gist_rainbow')
        # colors = [cm(1.*host_number/NUM_COLORS) for host_number in range(0, NUMBER_OF_RENO_HOSTS)]
        # ax1.set_prop_cycle(cycler('color', colors))
        # for host_number in range(0, NUMBER_OF_RENO_HOSTS):
            # ax1.plot(interval ,throughput[host_number], label = 'reno_' + str(host_number))
        # colors = [cm(1 - 1.*host_number/NUM_COLORS) for host_number in range(NUMBER_OF_RENO_HOSTS, NUM_COLORS)]
        # ax1.set_prop_cycle(cycler('color', colors))
        # for host_number in range(NUMBER_OF_RENO_HOSTS, NUMBER_OF_RENO_HOSTS + NUMBER_OF_VEGAS_HOSTS):
            # ax1.plot(interval ,throughput[host_number], label = 'vegas_' + str(host_number))

        ax1.plot(interval ,queue_size, label = 'queue_size')
        ax1.tick_params(axis='y')
        ax1.legend(loc=1)    
        plt.show()
    """


def create_sim_name(cwnd_algo_dict):
    name=''
    if len(cwnd_algo_dict)==0:
        return "WTF"
    for key, val in cwnd_algo_dict.items():
        name += "%d_%s_" %(val, key)
    return name[0:-1]

if __name__ == '__main__':
    # Simulation's parameters initializing:
    host_bw = 100
    host_delay = 10e3
    srv_bw = 500
    srv_delay = 20e3
    tcp_packet_size = 2806
    algo_dict = {}
    algo_dict['cubic']=3
    algo_dict['reno']=2
    algo_dict['vegas']=3

    queue_size = 2 * (
                srv_bw * srv_delay) / tcp_packet_size  # Rule of thumb: queue_size = (bw [Mbit/sec] * RTT [sec]) / size_of_packet.
    # Tell mininet to print useful information:
    setLogLevel('info')

    # bw is in Mbps, delay in msec, queue size in packets
    simulation_topology = SimulationTopology(algo_dict, host_bw=100, srv_bw=200,
                                             srv_delay=15000, rtr_queue_size=queue_size)
    simulation_name = create_sim_name(algo_dict)
    simulator = Iperf3Simulator(simulation_topology, simulation_name, 10)
    simulator.StartSimulation()
    tcp_stat = TcpdumpStatistics(simulator.port_algo_dict, )
    for filename in simulator.file_captures:
        tcp_stat.parse_dump_file(filename)
    tc_qlen_stat = TcQlenStatistics(simulator.rtr_q_filename)
    plotter:Plotter = Plotter(plot_file=os.path.join(simulator.res_dirname, 'Graphs.png'))
    plotter.create_throughput_plot(tcp_stat, tc_qlen_stat)
    plotter.create_ts_val_plot(tcp_stat)
    plotter.save_and_show()
