"""Custom topology example

Two directly connected switches plus a host for each switch:

   host --- switch --- switch --- host

Adding the 'topos' dict with a key/value pair to generate our newly defined
topology enables one to pass in '--topo=mytopo' from the command line.
"""
from logging import info

from mininet.link import TCLink
from mininet.node import Node
from mininet.topo import Topo



class LinuxRouter(Node):
    "A Node with IP forwarding enabled."
    def config(self, **params):
        super(LinuxRouter, self).config(**params)
        # Enable forwarding on the router
        info ('enabling forwarding on ', self)
        self.cmd('sysctl net.ipv4.ip_forward=1')

    def terminate(self):
        self.cmd('sysctl net.ipv4.ip_forward=0')
        super(LinuxRouter, self).terminate()

class RenoVSVegasTopo(Topo):
    def __init__(self, num_of_reno_hosts, num_of_vegas_hosts, host_bw = None, host_delay = None,
                 srv_bw = None, srv_delay = None, rtr_queue_size = None):
        self.num_of_reno_hosts = num_of_reno_hosts
        self.num_of_vegas_hosts = num_of_vegas_hosts
        self.host_bw = host_bw
        self.host_delay = host_delay
        self.srv_bw = srv_bw
        self.srv_delay = srv_delay
        self.rtr_queue_size = rtr_queue_size
        self.host_list = []

        self.as_addr_prefix = "10.0."


        # super should be called only after class member initialization.
        # The super constructor calls build.
        super(RenoVSVegasTopo, self).__init__()

    def add_host(self, algo, index):
        client_subnet_prefix = self.as_addr_prefix + str(index)
        host = self.addHost(algo + '_' + str(index),
                            ip=client_subnet_prefix + '.10/24',
                            defaultRoute="via " + client_subnet_prefix + '.1'
                            )
        self.host_list.append(host)

    def build(self, **_opts):
        # The router configuration:
        self.rtr = self.addNode('r', cls = LinuxRouter)

        # Reno hosts configuration:
        for host_number in range(0, self.num_of_reno_hosts):
            self.add_host("reno", host_number)
            self.addLink(self.host_list[host_number], self.rtr,
                         intfName1="host_int", # we keep the same if name to make tshark run easier
                         intfName2='r-reno_' + str(host_number),
                         params2={'ip': self.as_addr_prefix + str(host_number) + '.1/24'},
                         bw=self.host_bw, delay=self.host_delay
                         )
        #intfName1 = 'reno_' + str(host_number) + '-r',

        # Vegas hosts configuration:
        for host_number in range(self.num_of_reno_hosts, self.num_of_reno_hosts + self.num_of_vegas_hosts):
            self.add_host("vegas", host_number)
            self.addLink(self.host_list[host_number], self.rtr,
                         intfName1 = 'host_int',
                         intfName2 = 'r-vegas_' + str(host_number),
                         params2 = {'ip' : self.as_addr_prefix + str(host_number) + '.1/24'},
                         bw = self.host_bw,
                         delay = self.host_delay
                         )
        #intfName1 = 'vegas_' + str(host_number) + '-r',

        # The Server (the receiver) configuration:
        srv_addr = self.as_addr_prefix + str(self.num_of_reno_hosts + self.num_of_vegas_hosts) + '.10'
        self.srv = self.addHost('srv', ip = srv_addr + '/24',
                                defaultRoute = "via " + self.as_addr_prefix + str(self.num_of_reno_hosts + self.num_of_vegas_hosts) + '.1')
        self.addLink(self.srv, self.rtr, intfName1 = 'srv-r',
                     intfName2 = 'r-srv',
                     params2 = {
                         'ip' : self.as_addr_prefix + str(self.num_of_reno_hosts + self.num_of_vegas_hosts) + '.1/24',
                         'delay' : str(self.srv_delay)
                     },
                     bw = self.srv_bw,
                     max_queue_size=int(self.rtr_queue_size))

