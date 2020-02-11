"""Custom topology example

Two directly connected switches plus a host for each switch:

   host --- switch --- switch --- host

Adding the 'topos' dict with a key/value pair to generate our newly defined
topology enables one to pass in '--topo=mytopo' from the command line.
"""
from mininet.link import TCLink
from mininet.topo import Topo


class TwoSwitchIncastTopo(Topo):
    """A topology with two switches that are connected to each other.
     s1 conected to all clients and s2 connected to a server"""

    def __init__(self, k=2, bw=100):
        "k: number of clients"
        self.k = k
        self.bw = bw
        self.client_list = []

        # super should be called only after class member initialization.
        # The super constructor calls build.
        super(TwoSwitchIncastTopo, self).__init__()

    def build(self, **_opts):
        s1 = self.addSwitch('s1')
        s2 = self.addSwitch('s2')
        self.srv = self.addHost('my_server')
        self.addLink(s1, s2, bw=self.bw, cls = TCLink)
        self.addLink(self.srv, s2, bw=self.bw, cls = TCLink)
        for h in range(1, self.k+1):
            client = self.addHost('client%s' % h)
            self.addLink(client, s1, bw=self.bw, cls = TCLink)
            self.client_list.append(client)



