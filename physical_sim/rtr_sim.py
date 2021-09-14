#!/usr/bin/python3
import os
import shlex
import subprocess
import sys

from paramiko.client import SSHClient
from pssh.clients import ParallelSSHClient
from sismaot import getCredentials

user, password = getCredentials()
algo_list = ['reno', 'bbr', 'cubic']  # IMPORTANT! should be inline with online_filtering algo_dict
sim_name = sys.argv[1]
#max_num_of_bg_flows = 40
num_of_bg_flow_list = [0,5,10,20,25]
iperf_duration = 100
min_q_size, max_q_size, q_size_step = 100, 1001, 20
measured_flows_duration = len(num_of_bg_flow_list)*(iperf_duration+10)*int((max_q_size-min_q_size)/q_size_step)

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

# Run measured iperf3 flows
cmd = 'iperf3 -c 10.0.100.1 -p 5201 --cport 64501 -t %d -i 10 -C reno '%measured_flows_duration
client01.exec_command(cmd)
cmd = 'iperf3 -c 10.0.102.1 -p 5202 --cport 64502 -t %d -i 10 -C bbr'%measured_flows_duration
client23.exec_command(cmd)
cmd = 'iperf3 -c 10.0.103.1 -p 5203 --cport 64503 -t %d -i 10 -C cubic'%measured_flows_duration
client23.exec_command(cmd)


for packet_limit in range(min_q_size, max_q_size, q_size_step):
    # set the qdisc length
    cmd = 'sudo tc qdisc replace dev eno2 root pfifo limit %d ' % packet_limit
    cmds = shlex.split(cmd)
    subprocess.call(cmds)

    for  num_of_bg_flows in num_of_bg_flow_list:
        # Clean the previous round's clsact definitions, if any
        cmd = 'sudo ./clean_clsact.sh'
        cmds = shlex.split(cmd)
        subprocess.call(cmds)

        # Generate iperf3 background flows
        cmd = '~/bg_iperf.py  %d %d 5211 10.0.100.1 reno' % (num_of_bg_flows, iperf_duration)
        client01.exec_command(cmd)
        cmd = '~/bg_iperf.py  %d %d 5300 10.0.101.1 bbr' % (num_of_bg_flows, iperf_duration)
        client01.exec_command(cmd)
        cmd = '~/bg_iperf.py  %d %d 5400 10.0.102.1 cubic' % (num_of_bg_flows, iperf_duration)
        client23.exec_command(cmd)

        # Monitor the router and create raw data
        curr_sim_name = 'NumBG_%d_Queue_%d_%s' % (num_of_bg_flows*3, packet_limit, sim_name)
        cmd = 'sudo ../online_sim/tcp_smart_dump.py %s %d eno2 enp2s0f0 enp2s0f1 enp2s0f2 enp2s0f3' % (
        curr_sim_name, iperf_duration)
        cmds = shlex.split(cmd)
        subprocess.call(cmds)
