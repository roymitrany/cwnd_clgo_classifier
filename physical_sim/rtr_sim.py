#!/usr/bin/python3
import os
import shlex
import subprocess
import sys
import time
from random import randint

from paramiko.client import SSHClient
from pssh.clients import ParallelSSHClient
from sismaot import getCredentials

user, password = getCredentials()
algo_list = ['reno', 'bbr', 'cubic']
# algo_list = ['reno', 'bbr', 'cubic', 'vegas', 'htcp', 'bic']  # IMPORTANT! should be inline with online_filtering algo_dict
sim_name = sys.argv[1]
num_of_bg_flow_list = [0]
# num_of_bg_flow_list = [5, 10, 20, 25]
iperf_duration = 70
min_q_size, max_q_size, q_size_step = 1000, 6001, 1000
# list of algo permutations the number represents the location in algo_list.
# The location in the tuple determine for which sender the algo refers to.
# Example: (1,1,2) determines that senders 0 and 1 will user bbr algo, and sender 2 will use cubic algo.
# Altogether we have 9 permutation with 1 or 2 cwnd algos.
# l_of_t = [(0,1,2),(0,0,0),(1,1,1),(2,2,2),(3,3,3),(4,4,4)]
# l_of_t = [(0,1,2),(0,0,0),(1,1,1),(2,2,2), (0,0,1),(0,0,2), (1,1,2), (1,1,0), (2,2,0), (2,2,1)]
# l_of_t = [(0, 1, 2)]

client01 = SSHClient()
client01.load_host_keys(os.path.join(os.path.dirname(__file__), 'known_hosts'))
client01.connect('132.68.60.131', username=user, password=password)
client23 = SSHClient()
client23.load_host_keys(os.path.join(os.path.dirname(__file__), 'known_hosts'))
client23.connect('132.68.60.135', username=user, password=password)
router01 = SSHClient()
router01.load_host_keys(os.path.join(os.path.dirname(__file__), 'known_hosts'))
router01.connect('132.68.60.175', username=user, password=password)
servers_list = ['18.183.252.121', '3.101.103.141', '18.198.65.229']
# This file is the main loop for the router.
# It cleans clsact, sets qdisc limit length and starts tcp ebpf monitor
# It will stop the monitor before changing the limit, for stability purposes (hopefully)
# We use subprocess call, becaue all of the commands should complete before the next command executes


for packet_limit in range(min_q_size, max_q_size, q_size_step):
    # set the qdisc length on router01 server side interface
    cmd = 'sudo tc qdisc replace dev eno2 root pfifo limit %d ' % packet_limit
    router01.exec_command(cmd)
    for s1_algo in algo_list:
        for s2_algo in algo_list:
            for s3_algo in algo_list:
                # Run measured iperf3 flows
                curr_algo_list = [s1_algo, s2_algo, s3_algo]
                for server in servers_list:
                    algo = curr_algo_list[servers_list.index(server)]
                    # The source port gives a hint to the ML system about the CCA
                    src_port = 64501 + algo_list.index(algo) + 10*servers_list.index(server)
                    cmd = 'iperf3 -c %s --cport %d -t %d -i 10 -C %s' % (server, src_port, iperf_duration, algo)
                    print(cmd)
                    if servers_list.index(server) < len(algo_list) / 2:
                        curr_client = client01
                    else:
                        curr_client = client23
                    curr_client.exec_command(cmd)

                algo_abbr = curr_algo_list[0][0] + curr_algo_list[1][0] + curr_algo_list[2][0]
                curr_sim_name = 'NumBG_0_%s_Algo_Queue_%d_%s' % (algo_abbr, packet_limit, sim_name)

                # Run the smart dump command (different interfaces) remotely in router01
                cmd = 'cd ~/PycharmProjects/cwnd_clgo_classifier/online_sim && sudo ~/PycharmProjects/cwnd_clgo_classifier/online_sim/tcp_smart_dump.py %s %d enp6s0 enp1s0f0 enp1s0f1 ' \
                      'enp1s0f2 enp1s0f3 ' % ('r01_' + curr_sim_name, iperf_duration)
                print(cmd)
                stdin, stdout, stderr = router01.exec_command(cmd)
                exit_status = stdout.channel.recv_exit_status()  # Blocking call
                if exit_status == 0:
                    print(stdout)
                else:
                    print("Error", exit_status)
