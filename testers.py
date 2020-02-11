import os
import json
import re
from enum import Enum
from mininet.log import setLogLevel
from mininet.net import Mininet
from datetime import datetime, date, time
from mininet.node import OVSController
from mininet.util import pmonitor

from mytopo import TwoSwitchIncastTopo

class TcpAlgo(Enum):
    reno = 1
    cubic = 2
    vegas = 3
    westwood = 4
    bbr = 5


class IperfTester:
    def __init__(self, topo, test_name, seconds=10):
        """

        :param topo: The topology class to be used
        :param test_name: the results folder will contain the test name
        :param seconds: The test duration
        """
        # self.topo = TwoSwitchIncastTopo(k=5)
        self.topo = topo
        self.net = Mininet(topo, controller=OVSController)
        self.seconds = seconds
        self.test_name = test_name
        tn = datetime.now()
        time_str = str(tn.hour) + "_" + str(tn.minute) + "_" + str(tn.second)

        # str(ts).split(".")[0][-4:]
        self.results_filename = os.path.join(os.getcwd(), "results", test_name + "_" + time_str)

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
        for client in topo.client_list:
            self.res_dict[client] = []

        '''Interval dictionary stores the result per interval. so for m intervals and n clients it looks like this:
        For n clients running m intervals, the dict will look like this:
        int_1: [res_client_1, res_client2 ... res_client_n]
        int_2: [res_client_1, res_client2 ... res_client_n]
        .
        .
        .
        int_m: [res_client_1, res_client2 ... res_client_n]
        '''
        self.interval_dict = {}

    def start_test(self):
        self.net.start()
        srv = self.net.getNodeByName(self.topo.srv)
        srv_popen = srv.popen('iperf -s')
        srv_ip = srv.IP()
        popens = {}
        for client in self.topo.client_list:
            # Get server IP address somehow
            # cmd = 'iperf -c %s -t %d -i 1 > %s/iperf_%s.txt' % (srv_ip, seconds, dirname, client)
            cmd = 'iperf -c %s -t %d -i 1' % (srv_ip, self.seconds)
            popens[client] = (self.net.getNodeByName(client)).popen(cmd)

        # We need to wait util all commands are completed
        for client, line in pmonitor(popens, timeoutms=1000):
            if client:
                print '<%s>: %s' % (client, line),
            if client in self.res_dict:
                self.res_dict[client].append(line)
        # Kill the server's iperf -s
        srv.cmd('kill %iperf')
        self.net.stop()
        print self.res_dict

        # Save in file as JSON format
        with open(self.results_filename, 'wb') as outfile:
            json.dump(self.res_dict, outfile)

        outfile.close()

    def parse_results(self):
        with open(self.results_filename) as infile:
            data = json.load(infile)
            for k, v in data.items():
                # 0.0- 1.0 sec   745 KBytes  6.11 Mbits/sec\n'
                for line in v:
                    match_obj = re.match(r'.*\D(\d+\.\d+) sec.* \D(\d*\.?\d+) [a-zA-Z]?bits.*', line)
                    if match_obj:
                        end_interval = float(match_obj.group(1))
                        if str(end_interval) not in self.interval_dict:
                            self.interval_dict[str(end_interval)] = []
                        throughput = float(match_obj.group(2))
                        # we want the results in Mbps. If the results are in Kbps, divide by 1000
                        if re.search("Kbits/sec", line):
                            throughput = throughput / 1000
                        if re.search(" bits/sec", line):
                            throughput = throughput / 1000000

                        # add the result to the interval
                        self.interval_dict[str(end_interval)].append(throughput)

        # Print the interval dictionary
        for k, v in self.interval_dict.items():
            print k,
            print "---->>>>",
            print v

        infile.close()

    def change_tcp_cwnd_algo(self, client_idx, tcp_algo):
        """
        :param client_idx:
        :type int:
        :param tcp_algo: a member of TcpAlgo enum e.g. TcpAlgo.westwood
        :type int:
        :return:
        """
        # TODO: do we want to throw exceptions here?
        if client_idx >= self.topo.k:
            print "index bigger than num of servers"
            return

        client = topo.client_list[client_idx]
        cmd = 'echo %s > /proc/sys/net/ipv4/tcp_congestion_control' % tcp_algo.name
        self.net.getNodeByName(client).cmd(cmd)


if __name__ == '__main__':
    # Tell mininet to print useful information
    num_of_clients = 2
    setLogLevel('info')
    topo = TwoSwitchIncastTopo(num_of_clients)
    tester = IperfTester(topo, "kuku1", 20)
    tester.change_tcp_cwnd_algo(1, TcpAlgo.bbr)
    tester.start_test()
    tester.parse_results()

    # Calculate eoverall throughput
    for interval in sorted(tester.interval_dict.keys()):
        if len(tester.interval_dict[interval]) == num_of_clients:
            overall_thoughput = sum(tester.interval_dict[interval])
            print "Overall throughput in interval %s is %d" % (interval, overall_thoughput)
        else:
            print "interval %s got %d results" % (interval, len(tester.interval_dict[interval]))
