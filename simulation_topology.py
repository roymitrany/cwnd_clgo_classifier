"""
Congestion control reavluation classification topology:

   reno_host_i-----+
        .    |
        .    r ---- srv
        .    |
   vegas_host_i----+

"""
from logging import info
from dictionaries  import Dict

# from mininet.link import TCLink
from mininet.node import Node
from mininet.topo import Topo


class LinuxRouter(Node):
    "A Node with IP forwarding enabled."
    def config(self, **params):
        super(LinuxRouter, self).config(**params)
        # Enable forwarding on the router
        info('enabling forwarding on ', self)
        self.cmd('sysctl net.ipv4.ip_forward=1')

    def terminate(self):
        self.cmd('sysctl net.ipv4.ip_forward=0')
        super(LinuxRouter, self).terminate()


class SimulationTopology(Topo):
    def __init__(self, algo_dict, host_bw=None, host_delay=None, srv_bw=None, srv_delay=None, rtr_queue_size=None):
        self.algo_dict = algo_dict
        self.host_bw = host_bw
        self.host_delay = host_delay
        self.srv_bw = srv_bw
        self.srv_delay = srv_delay
        self.rtr_queue_size = rtr_queue_size
        self.host_list = []

        self.as_addr_prefix = "10.0."

        # super should be called only after class member initialization.
        # The super constructor calls build.
        super(SimulationTopology, self).__init__()

    def add_host(self, algo, index):
        client_subnet_prefix = self.as_addr_prefix + str(index)
        host = self.addHost(algo + '_' + str(index),
                            ip=client_subnet_prefix + '.10/24',
                            defaultRoute="via " + client_subnet_prefix + '.1'
                            )
        self.host_list.append(host)

    def build(self, **_opts):
        # The router configuration:
        self.rtr = self.addNode('r', cls=LinuxRouter)

        # Hosts configuration:
        host_number = 0
        for key, val in self.algo_dict.items():
            for h in range(0, val):
                self.add_host(key, host_number)
                self.addLink(self.host_list[host_number], self.rtr,
                             intfName1='%s_%s-r' % (key, str(host_number)),
                             intfName2='r-%s_%s' % (key, str(host_number)),
                             params2={'ip': self.as_addr_prefix + str(host_number) + '.1/24'},
                             bw=self.host_bw,
                             use_tbf=True,
                             delay=self.host_delay
                             )
                host_number += 1

        # The Server (the receiver) configuration Its index is the current host number defined in clients loop:
        srv_addr = self.as_addr_prefix + str(host_number) + '.10'
        self.srv = self.addHost('srv', ip=srv_addr + '/24',
                                defaultRoute="via " + self.as_addr_prefix + str(host_number) + '.1')
        self.addLink(self.srv, self.rtr,
                     intfName1='srv-r',
                     intfName2='r-srv',
                     params2={
                         'ip': self.as_addr_prefix + str(host_number) + '.1/24',
                         'delay': str(self.srv_delay)
                     },
                     bw=self.srv_bw,
                     use_tbf=True,
                     max_queue_size=int(self.rtr_queue_size)
                     )
