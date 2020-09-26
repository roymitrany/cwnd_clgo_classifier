#!/usr/bin/python3

import os
import json
import threading
import random
from pathlib import Path

from enum import Enum
from signal import SIGINT
from time import sleep
from mininet.link import TCLink
from mininet.log import setLogLevel
from mininet.net import Mininet
from datetime import datetime
from mininet.node import OVSController
from mininet.util import pmonitor

from simulation.simulation_topology import SimulationTopology
from simulation.single_connection_statistics import SingleConnStatistics
from simulation.tcpdump_statistics import TcpdumpStatistics
from simulation.tc_qdisc_statistics import TcQdiscStatistics
from simulation.graph_implementation import GraphImplementation


def create_csv(sim_obj, client, generate_graphs=False):
    print("calculating statistics for %s" % client)
    in_file = os.path.join(sim_obj.res_dirname, 'client_%s.txt' % client)
    out_file = os.path.join(sim_obj.res_dirname, 'server_%s.txt' % client)
    rtr_file = os.path.join(sim_obj.res_dirname, 'rtr_q.txt')
    graph_file_name = os.path.join(sim_obj.res_dirname, 'Conn_Graph_%s.png' % client)
    plot_title = client
    q_line_obj = SingleConnStatistics(in_file, out_file, rtr_file, graph_file_name, plot_title, generate_graphs)
    q_line_obj.conn_df.to_csv(os.path.join(sim_obj.res_dirname, 'single_connection_stat_%s.csv' % client))


class Iperf3Simulator:
    def __init__(self, simulation_topology, simulation_name, seconds=10):
        """
        :param simulation_topology: The topology class to be used
        :param simulation_name: the results folder will contain the test name
        :param seconds: The test duration
        """
        self.file_captures = []
        self.simulation_topology: SimulationTopology = simulation_topology
        self.net = Mininet(simulation_topology, controller=OVSController, link=TCLink, autoSetMacs=True)
        self.seconds = seconds
        self.simulation_name = simulation_name
        self.port_algo_dict = {}
        tn = datetime.now()
        time_str = str(tn.month) + "." + str(tn.day) + "." + str(tn.year) + "@" + str(tn.hour) + "-" + str(
            tn.minute) + "-" + str(tn.second)

        # Create results directory, with name includes num of clients for each algo, and time:
        self.res_dirname = os.path.join(os.getcwd(), "results", time_str + "_" + self.simulation_name)
        os.mkdir(self.res_dirname, 0o777)

        # Set results file names:
        self.iperf_out_filename = os.path.join(self.res_dirname, "iperf_output.txt")
        self.rtr_q_filename = os.path.join(self.res_dirname, "rtr_q.txt")

        # Create the simulation parameters file
        topo_param_filename = os.path.join(self.res_dirname, "topo_params.txt")
        param_file = open(topo_param_filename, 'w')
        param_dict = simulation_topology.to_dict()
        param_json = json.dumps(param_dict)
        param_file.write(param_json)
        param_file.close()

        # Create train csv file
        train_csv_filename = os.path.join(self.res_dirname, "train.csv")
        train_file = open(train_csv_filename, 'w')
        train_file.write("id,label\n")
        for host in self.simulation_topology.host_list:
            for algo in Algo:
                if algo.name in host:
                    train_file.write("%s,%d\n" % (host, algo.value))
        train_file.close()


    def SetCongestionControlAlgorithm(self, host, tcp_algo):
        """
        :param tcp_algo: a member of TcpAlgorithm enum e.g. TcpAlgorithm.westwood
        """
        cmd = 'echo %s > /proc/sys/net/ipv4/tcp_congestion_control' % tcp_algo
        self.net.getNodeByName(host).cmd(cmd)

    def process_results(self, generate_graphs=False, keep_dump_files=False):
        threads = list()
        for client in self.simulation_topology.host_list:
            x = threading.Thread(target=create_csv, args=(self, client))
            threads.append(x)
            x.start()

        for index, thread in enumerate(threads):
            thread.join()

        if generate_graphs:
            tcpdump_statistsics = TcpdumpStatistics(self.port_algo_dict)
            for filename in self.file_captures:
                tcpdump_statistsics.parse_tcpdump_file(filename)
            tc_qdisc_statistics = TcQdiscStatistics(self.rtr_q_filename)
            GraphImplementation(tcpdump_statistsics, tc_qdisc_statistics,
                                plot_file_name=os.path.join(self.res_dirname, 'Graphs.png'),
                                plot_fig_name="host_bw_%s_host_delay_%s_srv_bw_%s_srv_delay_%s_queue_size_%s.png" % (
                                    host_bw, host_delay, srv_bw, srv_delay, queue_size))
        if (keep_dump_files == False):
            for p in Path(self.res_dirname).glob("client_*.txt"):
                print(p)
                p.unlink()
            for p in Path(self.res_dirname).glob("server_*.txt"):
                print(p)
                p.unlink()

    def StartSimulation(self):
        self.net.start()
        # CLI(self.net)

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

            # Disable TSO for the client
            cmd = "ethtool -K %s-r tso off" % client
            self.net.getNodeByName(client).cmd(cmd)

        # Disable TSO for router
        cmd = "ethtool -K r-srv tso off"
        rtr.cmd(cmd)

        sleep(5)
        # Traffic generation loop:
        client_counter = 0
        for client in self.simulation_topology.host_list:
            cwnd_algo = client[0:client.find("_")]
            # cmd = 'iperf3 -c %s -t %d -p %d -C %s' % (srv_ip, self.seconds, 5201 + client_counter, self.congestion_control_algorithm[client_counter % 2])
            cmd = 'iperf3 -c %s -t %d -p %d -C %s' % (srv_ip, self.seconds, 5201 + client_counter, cwnd_algo)
            popens[client] = (self.net.getNodeByName(client)).popen(cmd)
            client_counter += 1

        # Gather statistics from the router:
        q_proc = rtr.popen('python tc_qdisc_implementation.py r-srv %s' % self.rtr_q_filename)

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


def clean_sim():
    cmd = "mn -c"
    os.system(cmd)


def create_sim_name(cwnd_algo_dict):
    name = ''
    if len(cwnd_algo_dict) == 0:
        return "WTF"
    for key, val in cwnd_algo_dict.items():
        name += "%d_%s_" % (val, key)
    return name[0:-1]


if __name__ == '__main__':
    # Simulation's parameters initializing:
    srv_delay = 5e3
    # tcp_packet_size = 2806
    Algo = Enum('Algo', 'reno bbr cubic')
    algo_dict = {}
    simulation_duration = 60  # seconds.
    # total_bw = max(host_bw * sum(algo_dict.itervalues()), srv_bw).

    # queue_size = 800  # 2 * (
    # srv_bw * total_delay) / tcp_packet_size  # Rule of thumb: queue_size = (bw [Mbit/sec] * RTT [sec]) / size_of_packet.
    # Tell mininet to print useful information:
    setLogLevel('info')
    # bw is in Mbps, delay in msec, queue size in packets:
    """for host_bw in range(70, 80, 10):
        for host_delay in range(4500, 4700, 200):
            for srv_bw in range(450, 470, 20):
                for queue_size in range(500, 1000, 100):"""
    # for host_bw in range(70, 120, 10):
    for host_delay in range(4500, 5500, 5):
        for srv_bw in range(450, 550, 5):
            for queue_size in range(500, 1000, 5):
                for algo in Algo:
                    algo_dict[algo.name] = random.randint(2, 4)
                total_delay = 2 * (host_delay + srv_delay)
                simulation_topology = SimulationTopology(algo_dict, host_delay=host_delay, host_bw=host_bw,
                                                         srv_bw=srv_bw,
                                                         srv_delay=srv_delay, rtr_queue_size=queue_size)
                simulation_name = create_sim_name(algo_dict)
                simulator = Iperf3Simulator(simulation_topology, simulation_name, simulation_duration)
                simulator.StartSimulation()
                simulator.process_results()
                clean_sim()
