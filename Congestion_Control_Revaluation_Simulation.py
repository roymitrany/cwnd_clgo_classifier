"""
Reno_VS_Vegas_simulation topology:

   reno_host_i-----+
        .    |
        .    r ---- srv
        .    |
   vegas_host_i----+

"""

from mininet.net import Mininet
from mininet.node import Node, OVSKernelSwitch, Controller, RemoteController
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.topo import Topo
from mininet.log import setLogLevel, info
from collections import OrderedDict

# Global hosts definitions:
NUMBER_OF_RENO_HOSTS = 5
NUMBER_OF_VEGAS_HOSTS = 5
reno_hosts = []
vegas_hosts = []

# Global IP parameters:
"""
RENO_ADDRESS = '10.0.1.' # Example for a complete reno IP: '10.0.1.10/24'
DEFAULT_RENO_ROUTE = 'via 10.0.1.1'
VEGAS_ADDRESS = '10.0.2.' # Example for a complete vegas IP: '10.0.2.10/24'
DEFAULT_VEGAS_ROUTE = 'via 10.0.2.1'
"""
HOST_ADDRESS = '10.0.' # Example for a complete host IP: '10.0.1.10/24'
DEFAULT_HOST_ROUTE = 'via 10.0.'
"""
SERVER_ADDRESS = '10.0.100.'
DEFAULT_SERVER_ROUTE = 'via 10.0.100.1'
"""
ADDRESSES_DICT = {}#OrderedDict()
ADDRESS_LIST = []
SERVER_PORT_NUMBER = 5430

# Global link's parameters:
QUEUE_SIZE = "200" #"1514"#"1514" # TCP packet size = 1514 bytes.
RTT = '110ms'		# r--srv link. when it was 100ms id didnt work!!!
BottleneckBW = 200 #0.8 # 100 mbit/s.
host_BW = 500 #80
host_RTT = "10ms"
"""
RenoBW = 80
VegasBW = 80
"""
# The following generates IP addresses from a subnet number and a host number
# ip(4,2) returns 10.0.4.2, and ip(4,2,24) returns 10.0.4.2/24
def ip(subnet,host,prefix=None):
    addr = '10.0.'+str(subnet)+'.' + str(host)
    if prefix != None: addr = addr + '/' + str(prefix)
    return addr

# For some examples we need to disable the default blocking of forwarding of packets with no reverse path
def rp_disable(host):
    ifaces = host.cmd('ls /proc/sys/net/ipv4/conf')
    ifacelist = ifaces.split()    # default is to split on whitespace
    for iface in ifacelist:
       if iface != 'lo': host.cmd('sysctl net.ipv4.conf.' + iface + '.rp_filter=0')

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
    def build(self, **_opts):
        # The router configuration:
        r  = self.addNode('r', cls = LinuxRouter)
        # Reno hosts configuration:
        for host_number in range(0, NUMBER_OF_RENO_HOSTS):
            reno_hosts.append(self.addHost('reno_' + str(host_number), ip= HOST_ADDRESS + str(host_number) + '.10/24', defaultRoute = DEFAULT_HOST_ROUTE + str(host_number) + '.1'))
            #reno_hosts.append(self.addHost('reno_' + str(host_number), ip= '10.0.0.10', defaultRoute = '10.0.0.2'))          
            ADDRESSES_DICT[HOST_ADDRESS + str(host_number) + '.10/24'] = 'reno_' + str(host_number)
            ADDRESS_LIST.append(HOST_ADDRESS + str(host_number) + '.10')
            #self.addLink('reno_' + str(host_number), r, intfName1 = 'reno_' + str(host_number) + '-r', intfName2 = 'r-reno_' + str(host_number), bw = RenoBW, params2 = {'ip' : RENO_ADDRESS + str(host_number) + '1/24'})
            self.addLink(reno_hosts[host_number], r, intfName1 = 'reno_' + str(host_number) + '-r', intfName2 = 'r-reno_' + str(host_number), params2 = {'ip' : HOST_ADDRESS + str(host_number) + '.1/24'}, bw = host_BW, delay = host_RTT)       
            #self.addLink(reno_hosts[host_number - 1], r, intfName1 = 'reno_' + str(host_number) + '-r', intfName2 = 'r-reno_' + str(host_number), bw = RenoBW, params2 = {'ip' : '10.0.0.2/24'})       
       # Vegas hosts configuration:
        for host_number in range(NUMBER_OF_RENO_HOSTS, NUMBER_OF_RENO_HOSTS + NUMBER_OF_VEGAS_HOSTS):
            vegas_hosts.append(self.addHost('vegas_' + str(host_number), ip= HOST_ADDRESS + str(host_number) + '.10/24', defaultRoute = DEFAULT_HOST_ROUTE + str(host_number) + '.1'))
            ADDRESSES_DICT[HOST_ADDRESS + str(host_number) + '.10/24'] = 'vegas_' + str(host_number)
            ADDRESS_LIST.append(HOST_ADDRESS + str(host_number) + '.10')
            #self.addLink('vegas_' + str(host_number), r, intfName1 = 'vegas_' + str(host_number) + '-r', intfName2 = 'r-vegas_' + str(host_number), bw = VegasBW, params2 = {'ip' : VEGAS_ADDRESS + str(host_number) + '1/24'})
            self.addLink(vegas_hosts[host_number - NUMBER_OF_RENO_HOSTS], r, intfName1 = 'vegas_' + str(host_number) + '-r', intfName2 = 'r-vegas_' + str(host_number), params2 = {'ip' : HOST_ADDRESS + str(host_number) + '.1/24'}, bw = host_BW, delay = host_RTT)
        # The Server (the receiver) configuration:
        SERVER_ADDRESS = HOST_ADDRESS + str(NUMBER_OF_RENO_HOSTS + NUMBER_OF_VEGAS_HOSTS) + '.10'
        srv = self.addHost('srv', ip = SERVER_ADDRESS + '/24', defaultRoute = DEFAULT_HOST_ROUTE + str(NUMBER_OF_RENO_HOSTS + NUMBER_OF_VEGAS_HOSTS) + '.1')
        self.addLink(srv, r, intfName1 = 'srv-r', intfName2 = 'r-srv', params2 = {'ip' : HOST_ADDRESS + str(NUMBER_OF_RENO_HOSTS + NUMBER_OF_VEGAS_HOSTS) + '.1/24'}, bw = BottleneckBW, delay = RTT)
        
    def RunSimulation(self):
       print()
        
def main():
    rtopo = RenoVSVegasTopo()
    net = Mininet(topo = rtopo,
                  link=TCLink,
                  autoSetMacs = True)  
    net.start()
    r = net['r']
    """
    if NUMBER_OF_RENO_HOSTS: # Apparently this interface (the first host configured) was not configured properly.
        r.cmd('ifconfig r-reno_1 10.0.1.1/24')
    else:
        r.cmd('ifconfig r-vegas_1 10.0.1.1/24')
    """
    #r.cmd('tc qdisc change dev r-srv handle 10: netem limit 10')
    #r.cmd('tc qdisc replace dev r-srv root bfifo limit {}'.format(QUEUE_SIZE))
    r.cmd('tc qdisc replace dev r-srv root pfifo limit {}'.format(QUEUE_SIZE))
    srv = net['srv']
    for node in [r, srv]: 
        node.cmd('/usr/sbin/sshd')
    for host in reno_hosts:
        net[host].cmd('/usr/sbin/sshd')
    for host in vegas_hosts:
        net[host].cmd('/usr/sbin/sshd')
        """
    h1.cmd('sudo echo 10.0.1.1 r >> /etc/hosts')
    h2.cmd('sudo echo 10.0.2.1 r >> /etc/hosts')
    srv.cmd('sudo echo 10.0.3.1 r >> /etc/hosts')
        """
    
    # Runing synchronized simulation:
    host_number = 0
    CLI(net)
    
    SERVER_ADDRESS = HOST_ADDRESS + str(NUMBER_OF_RENO_HOSTS + NUMBER_OF_VEGAS_HOSTS) + '.10'
    print(ADDRESSES_DICT)
    srv.cmd('xterm -e ./srv.sh {0} {1} &'.format(NUMBER_OF_RENO_HOSTS, NUMBER_OF_VEGAS_HOSTS))
    #srv.cmd('./srv.sh {0} {1} &'.format(NUMBER_OF_RENO_HOSTS, NUMBER_OF_VEGAS_HOSTS))
    
    for host in reno_hosts:
        #net[host].cmd('xterm -e ./host.sh {0} {1} {2} &'.format(SERVER_ADDRESS, SERVER_PORT_NUMBER + host_number, "reno"))
        net[host].cmd('./host.sh {0} {1} {2} {3} {4}&'.format(SERVER_ADDRESS, SERVER_PORT_NUMBER + host_number, "reno", host_BW, 0.01))
        host_number = host_number + 1
        #print(SERVER_PORT_NUMBER + host_number)
    for host in vegas_hosts:
        #net[host].cmd('xterm -e ./host.sh {0} {1} {2} &'.format(SERVER_ADDRESS, SERVER_PORT_NUMBER + host_number, "vegas"))
        net[host].cmd('./host.sh {0} {1} {2} {3} {4}&'.format(SERVER_ADDRESS, SERVER_PORT_NUMBER + host_number, "vegas",  host_BW, 0.01))
        host_number = host_number + 1

    #r.cmd('xterm -e ./r.sh {} {} &'.format(NUMBER_OF_RENO_HOSTS + NUMBER_OF_VEGAS_HOSTS, ADDRESS_LIST)) 
    r.cmd('./r.sh {0} {1} &'.format(NUMBER_OF_RENO_HOSTS + NUMBER_OF_VEGAS_HOSTS, ADDRESS_LIST)) 
    #r.cmd('tshark -i r-srv') 
    #CLI(net)
    #Reno_VS_Vegas_Graphs.waitForSimulationToFinish()
    
    #Reno_VS_Vegas_Graphs.printGraphs(ADDRESSES_DICT, NUMBER_OF_RENO_HOSTS, NUMBER_OF_VEGAS_HOSTS)

    CLI(net)
    net.stop()
setLogLevel('info')
main()
