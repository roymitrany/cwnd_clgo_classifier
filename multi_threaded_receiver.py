#!/usr/bin/python3

import copy
import re
import socket
import statistics
import time
from collections import OrderedDict
from sys import argv, exit
from threading import Thread, Timer, Lock

import matplotlib.pyplot as plt
from cycler import cycler

PORT_NUMBER = 5430
SERVER_BW = 0.8
SERVER_RTT = 0.11
NUMBER_OF_RENO_HOSTS = 0
NUMBER_OF_RENO_HOSTS = 0
MEASURING_INTERVAL = 1  # 0.01
HEADER_SIZE = 10
MIN_BUFFER_SIZE = 4608
throughput_dict_mutex = Lock()
queue_size_mutex = Lock()
active_sessions_mutex = Lock()


class ThreadedServer(object):
    def __init__(self, host, port):
        self.RTT = 0.0
        self.BW = 0.0
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.sock = socket.socket(socket.SOCK_STREAM, socket.AF_INET)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        #        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, int(MIN_BUFFER_SIZE / 2)) # socket size is MIN_BUFFER_SIZE. Otherwise(if smaller than that)- buffer size * 2.
        self.sock.bind((self.host, self.port))
        self.host_number = self.port - PORT_NUMBER
        # self.throughput_dict = OrderedDict()
        # throughput_dict[self.port] = 0

    def listen(self):
        Thread(target=self.listen_to_client).start()

    def listen_to_client(self):
        global THROUGHPUTS_DICT, QUEUE_SIZE, TOTAL_THROUGHPUT, start_time, active_sessions
        size = 1024
        self.sock.listen(5)
        client, address = self.sock.accept()
        while True:
            msg = client.recv(int(MIN_BUFFER_SIZE))
            if msg:
                throughput_dict_mutex.acquire()
                THROUGHPUTS_DICT[self.host_number] += len(msg)
                throughput_dict_mutex.release()
                # print(msg.decode("utf-8"))
                try:
                    # broadcasting_time = time.time() - float((re.findall(r'\S+', msg.decode("utf-8"))).pop())
                    # print(msg)
                    # print(self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)) # buff size
                    msg = re.findall(r'\S+', msg.decode("utf-8"))  # .pop()
                    # print(msg)
                    msg_number = float(msg[0])
                    # print(str(msg_number) + ' ' + str(msg_length))
                    broadcasting_time = time.time() - float(msg[3])
                    # TOTAL_THROUGHPUT[broadcasting_time] = (len(msg))# - TOTAL_THROUGHPUT[-1])
                    queue_size_mutex.acquire()
                    delay_in_queue = broadcasting_time - SERVER_RTT - self.RTT
                    QUEUE_SIZE.append(delay_in_queue * self.BW * 1e6)
                    # print(str(time.clock()) + ' ' + str(msg[3]))
                    queue_size_mutex.release()
                except:
                    continue
            else:
                client.close()
                # THROUGHPUTS_DICT[self.host_number] = self.throughput_dict
                active_sessions_mutex.acquire()
                active_sessions -= 1
                active_sessions_mutex.release()
                break


def StatsAnalysis(f):
    with open("/tmp/roy.txt", 'w') as f:

        global THROUGHPUTS_DICT, NOT_CUMMULATIVE_THROUGHPUTS_DICT, prev_throughput_dict, QUEUE_SIZE, start_time, active_sessions
        current_time = time.time() - start_time
        # print ('{}\t'.format(current_time), end="")
        for host_number in range(0, NUMBER_OF_RENO_HOSTS + NUMBER_OF_VEGAS_HOSTS):
            NOT_CUMMULATIVE_THROUGHPUTS_DICT[host_number] = (THROUGHPUTS_DICT[host_number] - prev_throughput_dict[
                host_number]) / MEASURING_INTERVAL
            # print(NOT_CUMMULATIVE_THROUGHPUTS_DICT[host_number], end="")
            f.write(str(NOT_CUMMULATIVE_THROUGHPUTS_DICT[host_number]) + ' ')
        queue_size_mutex.acquire()
        f.write(str(current_time) + ' ' + str(statistics.mean(QUEUE_SIZE) / 8) + '\n')  # byes and not bits.
        QUEUE_SIZE = [0]
        queue_size_mutex.release()
        # f.write(str(current_time)+' ')
        # subprocess.call('/media/sf_Thesis/mininet_project/srv_dualreceive.sh &', shell = True)
        # queue_size_mutex.acquire()
        prev_throughput_dict = copy.deepcopy(THROUGHPUTS_DICT)
        # queue_size_mutex.release()
        active_sessions_mutex.acquire()
        if (active_sessions == 0):
            active_sessions_mutex.release()
            exit()
        active_sessions_mutex.release()
        t = Timer(MEASURING_INTERVAL, StatsAnalysis)
        t.start()


def PrintGraphs():
    global QUEUE_SIZE, TOTAL_THROUGHPUT
    throughput = []
    with open("/tmp/roy.txt", 'r') as f:
        lines = f.readlines()
        for host_number in range(0, NUMBER_OF_RENO_HOSTS + NUMBER_OF_VEGAS_HOSTS):
            throughput.append([float(line.split()[host_number]) for line in lines[0:-1]])
        interval = [float(line.split()[NUMBER_OF_RENO_HOSTS + NUMBER_OF_VEGAS_HOSTS]) for line in lines[0:-1]]
        mean_queue_size = [float(line.split()[NUMBER_OF_RENO_HOSTS + NUMBER_OF_VEGAS_HOSTS + 1]) for line in
                           lines[0:-1]]
        f.close()
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('interval [s]')
    ax1.set_ylabel('throughput [bytes/s]')

    SEPERATION_SPREAD_SIZE = 2
    NUM_COLORS = SEPERATION_SPREAD_SIZE * (NUMBER_OF_RENO_HOSTS + NUMBER_OF_VEGAS_HOSTS)
    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(1. * host_number / NUM_COLORS) for host_number in range(0, NUMBER_OF_RENO_HOSTS)]
    ax1.set_prop_cycle(cycler('color', colors))
    # ax1.set_color_cycle([cm(1.*host_number/NUM_COLORS) for host_number in range(0, NUMBER_OF_RENO_HOSTS)])
    ax1.set_prop_cycle(cycler('color', colors))
    for host_number in range(0, NUMBER_OF_RENO_HOSTS):
        ax1.plot(interval, throughput[host_number], label='reno_' + str(host_number))  # , color=next(colors))
    # ax1.set_color_cycle([cm(1 - 1.*host_number/NUM_COLORS) for host_number in range(NUMBER_OF_RENO_HOSTS, NUM_COLORS)])
    colors = [cm(1 - 1. * host_number / NUM_COLORS) for host_number in range(NUMBER_OF_RENO_HOSTS, NUM_COLORS)]
    ax1.set_prop_cycle(cycler('color', colors))
    for host_number in range(NUMBER_OF_RENO_HOSTS, NUMBER_OF_RENO_HOSTS + NUMBER_OF_VEGAS_HOSTS):
        ax1.plot(interval, throughput[host_number], label='vegas_' + str(host_number))
    ax1.tick_params(axis='y')
    ax1.legend(loc=1)

    # ax1.plot(range(0, len(TOTAL_THROUGHPUT)),TOTAL_THROUGHPUT, label = 'vegas_' + str(host_number))
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('queue size [bytes]')  # we already handled the x-label with ax1
    # print (TOTAL_THROUGHPUT)
    try:
        ax2.plot(interval, mean_queue_size, label="queue_size")
        # ax2.plot(range(0, len(QUEUE_SIZE)),QUEUE_SIZE, label = "queue_size")
        # ax2.plot(range(0, len(TOTAL_THROUGHPUT)),TOTAL_THROUGHPUT, label = "queue_size")
    except:
        QUEUE_SIZE.pop()
        # ax2.plot(interval,QUEUE_SIZE, label = "QUEUE_SIZE", color="red")
        # ax2.plot(interval,TOTAL_THROUGHPUT, label = "QUEUE_SIZE", color="red")
    ax2.tick_params(axis='y', labelcolor="red")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax2.legend(loc=2)

    plt.show()


if __name__ == "__main__":
    global THROUGHPUTS_DICT, NOT_CUMMULATIVE_THROUGHPUTS_DICT, prev_throughput_dict, BROADCASTING_TIME_DICT, TOTAL_THROUGHPUT, QUEUE_SIZE, start_time, active_sessions
    THROUGHPUTS_DICT = OrderedDict()
    NOT_CUMMULATIVE_THROUGHPUTS_DICT = OrderedDict()
    prev_throughput_dict = OrderedDict()
    BROADCASTING_TIME_DICT = OrderedDict()

    QUEUE_SIZE = [0]
    TOTAL_THROUGHPUT = OrderedDict()

    start_time = time.time()
    if len(argv) > 1:
        NUMBER_OF_RENO_HOSTS = int(argv[1])
    if len(argv) > 2:
        NUMBER_OF_VEGAS_HOSTS = int(argv[2])
    if len(argv) > 3:
        SERVER_BW = int(argv[3])
    if len(argv) > 4:
        SERVER_RTT = int(argv[4])
    active_sessions = NUMBER_OF_RENO_HOSTS + NUMBER_OF_VEGAS_HOSTS -1
    for port_number in range(PORT_NUMBER, PORT_NUMBER + NUMBER_OF_RENO_HOSTS + NUMBER_OF_VEGAS_HOSTS):
        THROUGHPUTS_DICT[port_number - PORT_NUMBER] = 0
        NOT_CUMMULATIVE_THROUGHPUTS_DICT[port_number - PORT_NUMBER] = 0
        prev_throughput_dict[port_number - PORT_NUMBER] = 0
        ThreadedServer('', port_number).listen()
    StatsAnalysis()
    while (active_sessions):
        continue
    PrintGraphs()
