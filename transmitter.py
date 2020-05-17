#!/usr/bin/python3

import socket
import pickle
from sys import argv
import os
import time

default_host = "localhost"
portnum = 5431
host_BW = 80
host_RTT = 0.01
blockcount = 1000
cong_algorithm = 'reno'
HEADERSIZE = 10
MIN_BUFFER_SIZE = 4608

def talk():
    global portnum, blockcount, cong_algorithm, host_BW, host_RTT
    rhost = default_host
    if len(argv) > 1:
        blockcount = int(argv[1])
    if len(argv) > 2:
        rhost = argv[2]
    if len(argv) > 3:
        portnum = int(argv[3])
    if len(argv) > 4:
        cong_algorithm = argv[4]
    if len(argv) > 5:
        host_BW = argv[5]
    if len(argv) > 6:
        host_RTT = argv[6]
    print("Looking up address of " + rhost + "...", end="")
    try:
        dest = socket.gethostbyname(rhost)
    except socket.gaierror as mesg:
        errno,errstr=mesg.args
        print("\n   ", errstr);
        return;
    print("got it: " + dest)
    addr=(dest, portnum)
    #s = socket.socket(socket.SOCK_STREAM, socket.AF_INET)
    s = socket.socket()
    #IPPROTO_TCP = 6        	# defined in /usr/include/netinet/in.h
    TCP_CONGESTION = 13 	# defined in /usr/include/netinet/tcp.h
    cong = bytes(cong_algorithm, 'ascii')
    try:
       s.setsockopt(socket.IPPROTO_TCP, TCP_CONGESTION, cong)
    except OSError as mesg:
       errno, errstr = mesg.args
       print ('congestion mechanism {} not available: {}'.format(cong_algorithm, errstr))
       return
    s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, int(MIN_BUFFER_SIZE / 2)) # socket size is MIN_BUFFER_SIZE. Otherwise(if smaller than that)- buffer size * 2.
    res=s.connect_ex(addr)
    if res!=0: 
        print("connect to port ", portnum, " failed")
        return
    start_time = time.time()
    for i in range(blockcount):
        #s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, int(MIN_BUFFER_SIZE / 2)) # socket size is MIN_BUFFER_SIZE. Otherwise(if smaller than that)- buffer size * 2.
        #print(s.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF))
        broadcasting_time = time.time()
        msg = str(i) + ' ' + str(host_BW) + ' ' + str(host_RTT) + ' ' + str(broadcasting_time) + '\r\n'
        #msg = msg + ("{:>%d}"%(MIN_BUFFER_SIZE - len(msg,))).format('')
        msg = bytes(msg, "utf-8")
        msg = b' ' * (MIN_BUFFER_SIZE - len(msg)) + msg
        #msg = msg + ("{:>%d}"%(MIN_BUFFER_SIZE - len(msg) - len(str(broadcasting_time),))).format(broadcasting_time)
        #print(len(msg,))
        #print(s.send(msg))
        s.send(msg)
        #s.setsockopt(socket.IPPROTO_TCP, socket.TCP_CORK, 1)
        #s.send(bytes(msg, "utf-8"))
        #print(MIN_BUFFER_SIZE - len(str(broadcasting_time),)-len(str(host_BW),)-len(str(host_RTT),))
    s.close()
    print('total time: {} seconds'.format(time.time() - start_time))
        
talk()