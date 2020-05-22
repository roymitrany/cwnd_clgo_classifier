"""
Reno_VS_Vegas_simulation topology:

   reno_host_i-----+
        .    |
        .    r ---- srv
        .    |
   vegas_host_i----+

"""
from signal import SIGINT

from mininet.net import Mininet
from mininet.node import Node, OVSKernelSwitch, Controller, RemoteController
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.topo import Topo
from mininet.log import setLogLevel, info
from mininet.util import pmonitor

from collections import OrderedDict

# Global hosts definitions:
NUMBER_OF_RENO_HOSTS = 5
NUMBER_OF_VEGAS_HOSTS = 5
reno_hosts = []
vegas_hosts = []

# Global IP parameters:
AS_ADDR_PREFIX = '10.0.' # Example for a complete host IP: '10.0.1.10/24'
DEFAULT_HOST_ROUTE = 'via 10.0.'

ADDRESSES_DICT = {}#OrderedDict()
ADDRESS_LIST = []
SERVER_PORT_NUMBER = 5430

# Global link's parameters:
QUEUE_SIZE = "109" #"1514"#"1514" # TCP packet size = 1514 bytes.
RTT = '110ms'		# r--srv link. when it was 100ms id didnt work!!!
BottleneckBW = 1000 #0.8 # 100 mbit/s.
host_BW = 500
host_RTT = "1ms"

# The following generates IP addresses from a subnet number and a host number
# ip(4,2) returns 10.0.4.2, and ip(4,2,24) returns 10.0.4.2/24
def ip(subnet, host, prefix_len=None):
    addr = AS_ADDR_PREFIX+str(subnet)+'.' + str(host)
    if prefix_len != None: addr = addr + '/' + str(prefix_len)
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
        r = self.addNode('r', cls = LinuxRouter)

        # Reno hosts configuration:
        for host_number in range(0, NUMBER_OF_RENO_HOSTS):
            reno_hosts.append(self.addHost('reno_' + str(host_number),
                                           ip=AS_ADDR_PREFIX + str(host_number) + '.10/24',
                                           defaultRoute =DEFAULT_HOST_ROUTE + str(host_number) + '.1'
                                           )
                              )
            ADDRESSES_DICT[AS_ADDR_PREFIX + str(host_number) + '.10/24'] = 'reno_' + str(host_number)
            ADDRESS_LIST.append(AS_ADDR_PREFIX + str(host_number) + '.10')
            self.addLink(reno_hosts[host_number], r,
                         intfName1 = 'reno_' + str(host_number) + '-r',
                         intfName2 = 'r-reno_' + str(host_number),
                         params2 = {'ip' : AS_ADDR_PREFIX + str(host_number) + '.1/24'},
                         bw = host_BW, delay = host_RTT
                         )

       # Vegas hosts configuration:
        for host_number in range(NUMBER_OF_RENO_HOSTS, NUMBER_OF_RENO_HOSTS + NUMBER_OF_VEGAS_HOSTS):
            vegas_hosts.append(self.addHost('vegas_' + str(host_number),
                                            ip=AS_ADDR_PREFIX + str(host_number) + '.10/24',
                                            defaultRoute =DEFAULT_HOST_ROUTE + str(host_number) + '.1'
                                            )
                               )
            ADDRESSES_DICT[AS_ADDR_PREFIX + str(host_number) + '.10/24'] = 'vegas_' + str(host_number)
            ADDRESS_LIST.append(AS_ADDR_PREFIX + str(host_number) + '.10')
            self.addLink(vegas_hosts[host_number - NUMBER_OF_RENO_HOSTS], r,
                         intfName1 = 'vegas_' + str(host_number) + '-r',
                         intfName2 = 'r-vegas_' + str(host_number),
                         params2 = {'ip' : AS_ADDR_PREFIX + str(host_number) + '.1/24'},
                         bw = host_BW,
                         delay = host_RTT
                         )
        # The Server (the receiver) configuration:
        SERVER_ADDRESS = AS_ADDR_PREFIX + str(NUMBER_OF_RENO_HOSTS + NUMBER_OF_VEGAS_HOSTS) + '.10'
        srv = self.addHost('srv', ip = SERVER_ADDRESS + '/24',
                           defaultRoute = DEFAULT_HOST_ROUTE + str(NUMBER_OF_RENO_HOSTS + NUMBER_OF_VEGAS_HOSTS) + '.1')
        s0 = self.addSwitch('s0')
        self.addLink(s0, r, intfName1 = 'srv-r',
                     intfName2 = 'r-srv',
                     params2 = {'ip' : AS_ADDR_PREFIX + str(NUMBER_OF_RENO_HOSTS + NUMBER_OF_VEGAS_HOSTS) + '.1/24'},
                     bw = BottleneckBW,
                     delay = RTT,
                     max_queue_size=int(QUEUE_SIZE))

        self.addLink(s0, srv, cls=TCLink)

    def RunSimulation(self):
       print()
        
def main():
    rtopo = RenoVSVegasTopo()
    net = Mininet(topo = rtopo,
                  link=TCLink,
                  autoSetMacs = True)  
    net.start()
    # We will always keep this command here in case we want to xterm hosts during test "exit" command
    # from CLI prompt will let the program continue to run
    #CLI(net)
    r = net['r']
    srv = net['srv']
    popens = {}

    # Runing synchronized simulation:
    host_number = 0

    SERVER_ADDRESS = AS_ADDR_PREFIX + str(NUMBER_OF_RENO_HOSTS + NUMBER_OF_VEGAS_HOSTS) + '.10'
    cmd = './srv.sh {0} {1}'.format(NUMBER_OF_RENO_HOSTS, NUMBER_OF_VEGAS_HOSTS)
    popens[srv] = srv.popen(cmd)
    #ret = srv.cmd()
    #r.cmd('tshark -i r-srv -w /tmp/test.pcap -F libpcap&')
    for host in reno_hosts:
        net[host].cmd('./host.sh {0} {1} {2} {3} {4}&'.format(SERVER_ADDRESS, SERVER_PORT_NUMBER + host_number, "reno", host_BW, 0.01))
        host_number = host_number + 1
    for host in vegas_hosts:
        net[host].cmd('./host.sh {0} {1} {2} {3} {4}&'.format(SERVER_ADDRESS, SERVER_PORT_NUMBER + host_number, "vegas",  host_BW, 0.01))
        host_number = host_number + 1
    r.cmd('./r.sh {0} {1} &'.format(NUMBER_OF_RENO_HOSTS + NUMBER_OF_VEGAS_HOSTS, ADDRESS_LIST))
    q_proc = r.popen('python queue_len_poller.py r-srv /tmp/sigroy.txt')
    for server, line in pmonitor(popens): #TODO check efficiency
        if server:
            continue
    # Stop the queue monitor in the router
    q_proc.send_signal(SIGINT)
    #CLI(net) # Closing CLI prompt that enables us to save wireshark results if needed
    net.stop()
setLogLevel('info')
main()
