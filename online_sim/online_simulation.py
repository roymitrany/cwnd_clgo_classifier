#!/usr/bin/python3

import os
import re
import shutil
import threading
from pathlib import Path

from enum import Enum
from random import random, randint
from signal import SIGINT
from subprocess import Popen
from time import sleep, time

import numpy
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import setLogLevel
from mininet.net import Mininet
from datetime import datetime
import time
from mininet.node import OVSController
from mininet.util import pmonitor

from simulation.simulation_topology import SimulationTopology

class AlgoStreams:
    def __init__(self, measured_dict, unmeasured_dict):
        self.measured_dict = measured_dict
        self.unmeasured_dict = unmeasured_dict


class Iperf3Simulator:
    def __init__(self, simulation_topology, simulation_name, seconds=10, iperf_start_after=0,
                 background_noise=0, interval_accuracy=1, iteration=0):
        """
        :param interval_accuracy:
        :param simulation_topology: The topology class to be used
        :param simulation_name: the results folder will contain the test name
        :param seconds: The test duration
        :param iperf_start_after: iperf will start the after random time up to this value in ms
        :param tick_interval: tick duration. effects qdisc sample interval, background noise generation
        :param background_noise: Amount of packets per tick to send as background noise
        """
        self.file_captures = []
        self.simulation_topology: SimulationTopology = simulation_topology
        self.iperf_start_after: int = iperf_start_after
        self.net = Mininet(simulation_topology, controller=OVSController, link=TCLink, autoSetMacs=True)
        self.seconds = seconds
        self.simulation_name = simulation_name
        self.port_algo_dict = {}
        self.background_noise = background_noise
        self.interval_accuracy = interval_accuracy

    def SetCongestionControlAlgorithm(self, host, tcp_algo):
        """
        :param tcp_algo: a member of TcpAlgorithm enum e.g. TcpAlgorithm.westwood
        """
        cmd = 'echo %s > /proc/sys/net/ipv4/tcp_congestion_control' % tcp_algo
        self.net.getNodeByName(host).cmd(cmd)

    def start_simulation(self):
        self.net.start()

        srv = self.net.getNodeByName(self.simulation_topology.srv)
        srv_ip = srv.IP()
        popens = {}
        srv_procs = []
        rtr = self.net.getNodeByName(self.simulation_topology.rtr1)
        self.net['r1'].cmd("ip route add 10.1.0.0/24 via 10.100.0.2 dev r1-r2")

        for i in range(100):
            self.net['r2'].cmd("ip route add 10.0." + str(i) +".0/24 via 10.100.0.1 dev r2-r1")

        # Generate background noise
        noise_gen = self.net.getNodeByName(self.simulation_topology.noise_gen)

        if self.background_noise > 0:
            noise_gen.popen('python noise_generator.py %s %s' % (srv_ip, self.background_noise))
        client_counter = 0
        intf_name_str = ""
        #CLI(self.net)

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

            inbound_interface_name = "r1-%s" % client
            intf_name_str += inbound_interface_name
            intf_name_str += " "
            client_counter += 1

            # Disable TSO for the client
            cmd = "ethtool -K %s-r1 tso off" % client
            self.net.getNodeByName(client).cmd(cmd)

        # Run the online_sim command with all the interfaces. Server interface should always be the first one!
        ebpf_cmd = os.path.join(Path(os.getcwd()).parent, "online_sim", "tcp_smart_dump.py")
        cmd = "%s %d %s %s %s&>debug_files/%s_rtr_ebpf_out.txt" % (
            ebpf_cmd, simulation_duration, self.simulation_name, "r1-r2", intf_name_str, time.time())
        print(cmd)
        rtr_ebpf_proc = rtr.popen(cmd, shell=True)

        # Disable TSO for router
        cmd = "ethtool -K r2-srv tso off"
        rtr.cmd(cmd)

        sleep(2)
        # Traffic generation loop:
        client_counter = 0
        for client in self.simulation_topology.host_list:
            cwnd_algo = client[0:client.find("_")]
            start_after = randint(0, self.iperf_start_after) / 1000
            if client in self.simulation_topology.measured_host_list:
                client_port = 64501 + list(measured_dict.keys()).index(cwnd_algo)  # Set client port according to algo. hint for csv file names
            else:
                client_port = 64499

            cmd = 'sleep %f && iperf3 -c %s -t %d -p %d --bind 10.0.%d.10 --cport %d -C ' \
                  '%s>debug_files/ipuff_debig_%s.txt' % (
                      start_after, srv_ip, self.seconds, 5201 + client_counter, client_counter, client_port, cwnd_algo,
                      client)
            print("sleeeeeeeeeeeeeeeeeeeeeeping %s " % cmd)
            popens[client] = (self.net.getNodeByName(client)).popen(cmd, shell=True)
            client_counter += 1

        for client, line in pmonitor(popens, timeoutms=1000):
            if client:
                print('<%s>: %s' % (client, line), )
        print("--------------------after monitoring " + str(len(popens)))
        # Kill the server's iperf -s processes, the router's queue monitor and tcpdumps:
        rtr_ebpf_proc.send_signal(SIGINT)
        sleep(2)
        for process in srv_procs:
            process.send_signal(SIGINT)

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


def arrange_res_files():
    des_res_dir = os.path.join(Path(os.getcwd()).parent, "classification_data", "75_background_flows")
    curr_root_dir = os.path.join(Path(os.getcwd()).parent, "classification_data", "online")
    result_files = list(Path(curr_root_dir).rglob("*_6450[0-9]_*"))
    count = 0
    for res_file in result_files:
        search_obj = re.search(r'[0-9]+_[0-9]+_(6450[0-9])_[0-9]+_52[0-9][0-9].csv', str(res_file))
        if not search_obj:
            continue
        port = int(search_obj.group(1))
        port_offset = port-64501
        curr_algo = list(measured_dict.keys())[port_offset]
        file_new_name = 'single_connection_stat_%s_%d.csv' % (curr_algo, count)
        os.rename(res_file, os.path.join(os.path.dirname(res_file), file_new_name))
        count += 1

    # Move the files to final destination
    file_names = os.listdir(curr_root_dir)
    for file_name in file_names:
        # shutil.move(folderr, des_res_dir)
        shutil.move(os.path.join(curr_root_dir, file_name), des_res_dir)


if __name__ == '__main__':
    # interval accuracy: a number between 0 to 3. For value n, the accuracy will be set to 1/10^n
    interval_accuracy = 3

    # Simulation's parameters initializing:
    measured_dict = {}
    unmeasured_dict = {}
    simulation_duration = 70  # 60 # 80 # 120  # seconds.
    # total_bw = max(host_bw * sum(algo_dict.itervalues()), srv_bw).

    # queue_size = 800  # 2 * (
    # srv_bw * total_delay) / tcp_packet_size  # Rule of thumb: queue_size = (bw [Mbit/sec] * RTT [sec]) / size_of_packet.
    # Tell mininet to print useful information:
    setLogLevel('info')
    # bw is in Mbps, delay in msec, queue size in packets:

    background_noise = 0
    host_delay = 25 #2.5
    srv_delay = 25 #2.5
    iteration = 0

    host_bw = 50
    srv_bw = 50
    queue_size = 500
    for srv_bw in numpy.linspace(50, 100, 5):
        for host_bw in numpy.linspace(srv_bw, srv_bw + 100, 5):
            # for queue_size in numpy.linspace(100, 1000, 10):
            for lcount in range(10):
                # for srv_bw in range(10, 100, 20):
                # for queue_size in range(100, 1000, 100):
                measured_dict['reno'] = 1
                measured_dict['bbr'] = 1
                measured_dict['cubic'] = 1
                unmeasured_dict['reno'] = 25
                unmeasured_dict['bbr'] = 25
                unmeasured_dict['cubic'] = 25
                algo_streams = AlgoStreams(measured_dict, unmeasured_dict)
                # algo_dict['bbr']=10
                total_delay = 2 * (host_delay + srv_delay)
                simulation_topology = SimulationTopology(algo_streams, host_delay=host_delay, host_bw=host_bw,
                                                         srv_bw=srv_bw,
                                                         srv_delay=srv_delay, rtr_queue_size=queue_size)
                simulation_name = create_sim_name(measured_dict)
                iteration += 1
                simulator = Iperf3Simulator(simulation_topology, simulation_name, simulation_duration,
                                            iperf_start_after=2,
                                            background_noise=background_noise,
                                            interval_accuracy=interval_accuracy, iteration=iteration)
                # iperf_start_after=500, background_noise=100)
                simulator.start_simulation()
                arrange_res_files()
                clean_sim()
