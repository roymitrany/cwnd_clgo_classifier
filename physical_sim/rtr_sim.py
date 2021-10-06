#!/usr/bin/python3
import os
import shlex
import subprocess
import sys
from random import randint

from paramiko.client import SSHClient
from pssh.clients import ParallelSSHClient
from sismaot import getCredentials

user, password = getCredentials()
algo_list = ['reno', 'bbr', 'cubic']  # IMPORTANT! should be inline with online_filtering algo_dict
sim_name = sys.argv[1]
#max_num_of_bg_flows = 40
num_of_bg_flow_list = [5,10,20,25]
iperf_duration = 100
min_q_size, max_q_size, q_size_step = 2000, 15001, 1000
#list of algo permutations the number represents the location in algo_list.
# The location in the tuple determine for which sender the algo refers to.
# Example: (1,1,2) determines that senders 0 and 1 will user bbr algo, and sender 2 will use cubic algo.
# Altogether we have 9 permutation with 1 or 2 cwnd algos.
#l_of_t = [(0,1,2)]
l_of_t = [(0,1,2),(0,0,0),(1,1,1),(2,2,2), (0,0,1),(0,0,2), (1,1,2), (1,1,0), (2,2,0), (2,2,1)]
measured_flows_duration = len(num_of_bg_flow_list)*(iperf_duration+10)*int((max_q_size-min_q_size)/q_size_step)*len(l_of_t)

client01 = SSHClient()
client01.load_host_keys(os.path.join(os.path.dirname(__file__), 'known_hosts'))
client01.connect('132.68.60.206', username=user, password=password)
client23 = SSHClient()
client23.load_host_keys(os.path.join(os.path.dirname(__file__), 'known_hosts'))
client23.connect('132.68.60.131', username=user, password=password)

# This file is the main loop for the router.
# It cleans clsact, sets qdisc limit length and starts tcp ebpf monitor
# It will stop the monitor before changing the limit, for stability purposes (hopefully)
# We use subprocess call, becaue all of the commands should complete before the next command executes


for packet_limit in range(min_q_size, max_q_size, q_size_step):
    # set the qdisc length
    cmd = 'sudo tc qdisc replace dev eno2 root pfifo limit %d ' % packet_limit
    cmds = shlex.split(cmd)
    subprocess.call(cmds)

    for  num_of_bg_flows in num_of_bg_flow_list:
        for t in l_of_t:
            # Run measured iperf3 flows
            slp = randint(0, 2)
            cmd = 'sleep %d && iperf3 -c 10.0.100.1 -p 5201 --cport 64501 -t %d -i 10 -C reno ' % (slp, (iperf_duration-slp-5))
            client01.exec_command(cmd)
            slp = randint(0, 2)
            cmd = 'sleep %d && iperf3 -c 10.0.102.1 -p 5202 --cport 64502 -t %d -i 10 -C bbr' % (slp, (iperf_duration-slp-5))
            client23.exec_command(cmd)
            slp = randint(0, 2)
            cmd = 'sleep %d && iperf3 -c 10.0.103.1 -p 5203 --cport 64503 -t %d -i 10 -C cubic' % (slp, (iperf_duration-slp-5))
            client23.exec_command(cmd)

            # Clean the previous round's clsact definitions, if any
            cmd = 'sudo ./clean_clsact.sh'
            cmds = shlex.split(cmd)
            subprocess.call(cmds)

            # Generate iperf3 background flows
            cmd = '~/bg_iperf.py  %d %d 5211 10.0.100.1 %s' % (num_of_bg_flows, iperf_duration,algo_list[t[0]])
            client01.exec_command(cmd)
            cmd = '~/bg_iperf.py  %d %d 5300 10.0.101.1 %s' % (num_of_bg_flows, iperf_duration,algo_list[t[1]])
            client01.exec_command(cmd)
            cmd = '~/bg_iperf.py  %d %d 5400 10.0.102.1 %s' % (num_of_bg_flows, iperf_duration,algo_list[t[2]])
            client23.exec_command(cmd)

            # Monitor the router and create raw data
            # create abbreviation string e.g. ccr for cubic cubic reno. in case of 60 bg flows it will mean
            # 40 flows of cubic, 20 for reno and 0 for bbr
            algo_abbr = algo_list[t[0]][0]+algo_list[t[1]][0]+algo_list[t[2]][0]
            curr_sim_name = 'NumBG_%d_Algo_%s_Queue_%d_%s' % (num_of_bg_flows*3, algo_abbr, packet_limit, sim_name)
            cmd = 'sudo ../online_sim/tcp_smart_dump.py %s %d eno2 enp2s0f0 enp2s0f1 enp2s0f2 enp2s0f3' % (
            curr_sim_name, iperf_duration)
            cmds = shlex.split(cmd)
            subprocess.call(cmds)
