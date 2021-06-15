#!/home/another/PycharmProjects/cwnd_clgo_classifier/venv/bin/python3
# Copyright (c) PLUMgrid, Inc.
# Licensed under the Apache License, Version 2.0 (the "License")
import os
from pathlib import Path
import socket

print("startinggg tcp_smart_dump, before inport")
import sys
print (sys.path)
import sys
sys.path.insert(0,'..')
import subprocess
import traceback
from os import popen
import shlex
from signal import SIGINT

from bcc import BPF
from pyroute2 import IPRoute
import sys
import time
import pandas as pd
from ctypes import c_int, c_uint32, c_uint64
from simulation.single_connection_statistics import OnlineSingleConnStatistics

# TODO: get interval accuracy as parameter
interval_accuracy = 3
debug_file = open("ebpf_debug.txt", 'w')
print("starting tcp_smart_dump", file=debug_file)
traffic_duration = int(sys.argv[1])
intf_list = sys.argv[2:]
inft_index_list = []
ipr = IPRoute()
interval_duration = 6

num_of_samples = int(traffic_duration / interval_duration)
server_intf_index = -1
res_dir = os.path.join(Path(os.getcwd()).parent, "classification_data", "csvs")
try:
    b = BPF(src_file="tcp_smart_dump.c", debug=0)
    fn = b.load_func("handle_egress", BPF.SCHED_CLS)
    sniff_mode_table = b.get_table('sniff_mode')
    pkt_count_table = b.get_table('pkt_count')
    pkt_out_count_table = b.get_table('pkt_out_count')
    debug_val_table = b.get_table('debug_val')
    start_time_table = b.get_table('start_time')
    pkt_array = b.get_table('pkt_array')
    pkt_array_ext = b.get_table('pkt_array_ext')

    debug_file.write("Start of program!!!!?\n")
    debug_file.flush()
    ipr = IPRoute()

    # handle server facing interface first, read egress traffic only
    intf = intf_list[0]
    server_intf_index = ipr.link_lookup(ifname=intf)[0]
    inft_index_list.append(server_intf_index)
    print("server interface %s is: %d" % (intf, server_intf_index), file=debug_file)
    ipr.tc("add", "clsact", server_intf_index)

    # egress
    #ipr.tc("add-filter", "bpf", server_intf_index, ":1", fd=fn.fd, name=fn.name,
    #       parent="ffff:fff3", classid=1, direct_action=True)
    #debug_file.write("Egress BPF Filter Added for interface intf!!!!\n")
    #debug_file.flush()
    ffilter = b.load_func("out_filter", BPF.SOCKET_FILTER)
    BPF.attach_raw_socket(ffilter, sys.argv[2])

    # frontend part
    socketfd = ffilter.sock

    sockobj = socket.fromfd(socketfd, socket.AF_PACKET, socket.SOCK_RAW, socket.IPPROTO_IP)
    sockobj.setblocking(True)

    # ingress
    for intf in intf_list[1:]:
        intf_index = ipr.link_lookup(ifname=intf)[0]
        inft_index_list.append(intf_index)
        print("if %s is: %d" % (intf, intf_index), file=debug_file)
        ipr.tc("add", "clsact", intf_index)

        # ingress
        debug_file.write('adding  ingress filter for interface: %d\n' % intf_index)
        ipr.tc("add-filter", "bpf", intf_index, ":1", fd=fn.fd, name=fn.name,
               parent="ffff:fff2", classid=1, direct_action=True)

    try:
        for jj in range(num_of_samples):

            # Gather statistics from the router:
            q_disc_file = os.path.join(res_dir, "%d_qdisc.csv" % int(time.time()))
            #q_disc_file = "/home/user/csvs/%d_qdisc.csv" % int(time.time())
            q_disc_cmd = os.path.join(Path(os.getcwd()).parent, "simulation", "tc_qdisc_implementation.py")
            command_line = 'python3 %s %s %s %d'% (q_disc_cmd, sys.argv[2], q_disc_file, interval_accuracy)
            args = shlex.split(command_line)
            q_proc = subprocess.Popen(args)
            start_capture_epoch_time = time.time()
            sniff_mode_table[0] = c_uint32(1)
            start_time_table[0] = c_uint64(0) # reset start time
            time.sleep(interval_duration)
            sniff_mode_table[0] = c_uint32(0)
            q_proc.send_signal(SIGINT)
            packets_count = pkt_count_table[0].value
            packet_out_count = pkt_out_count_table[0].value
            debug_val = debug_val_table[0].value
            debug_file.write('before dict iteration loop ==============' + str(time.time()) + "\n")
            debug_file.flush()
            df_dict = {}
            # for key, pkt in pkt_array.items_lookup_and_delete_batch(): !!! not supported in Ubuntu 18.04
            for key, pkt in pkt_array.items():
                pkt_ext = pkt_array_ext.get(key)
                if (pkt_ext == None):
                    continue
                intf_index = key.ifindex
                conn_index = "%s_%s_%s_%s" % (pkt.src_ip, pkt.src_port, pkt.dst_ip, pkt.dst_port)
                # output_file.write('connection index is ' + conn_index + "\n")
                # output_file.flush()
                # print('connection index is ' + conn_index)
                # print ('if index is %s' %ifindex)
                # print ('packet length is %s' %pkt.length)

                # TODO: replace this extremely inefficient code
                if not conn_index in df_dict:
                    # Init ingress list and egres list
                    df_dict[conn_index] = {}

                if not intf_index in df_dict[conn_index]:
                    df_dict[conn_index][intf_index] = []

                # Insert to ingress list if if_index1 and egress if if_index2
                df_dict[conn_index][intf_index].append(
                    [conn_index, pkt.timestamp, pkt.length, pkt_ext.tsval, pkt_ext.seq_num])

            debug_file.write('num of captured packets: %d\n' % (packets_count))
            debug_file.write('num of captured packets on the way to server: %d\n' % (packet_out_count))
            debug_file.write('debug_vallj: %d\n' % (debug_val))
            debug_file.flush()
            # Loop on all TCP connections.
            for conn_index in df_dict.keys():
                ssrv_df = None
                clnt_df = None
                "For each connection we can expect 2 interfaces, one ingress and one egress traffic"
                "For now, we assume that there is only one egress port (server facing) in the router"
                for intf_index in df_dict[conn_index]:
                    debug_file.write('-----%screating dataframe \n' % str(time.time()))
                    debug_file.flush()
                    df = pd.DataFrame(df_dict[conn_index][intf_index],
                                      columns=['conn_index', 'date_time', 'length', 'ts_val', 'seq_num'])
                    df.index.name = "num"

                    # Get the capture start time, which is the number of nanoseconds from boot time until capture
                    # started.
                    # Then subtract the start time from packet capture time, which is the  number of nanoseconds from
                    # boot time until the packet was captured
                    # Add this result to epoch time in nanos at capture start time
                    # the result is the epoch time in nanos when the packet was captured.
                    start_time = start_time_table[0].value
                    # TODO: This doen not make any sence !!!! but the second formula seems to work
                    # Check in reasonable working hours
                    # df['date_time'] = df['date_time']-start_time+start_capture_time_nanos
                    debug_file.write('------packet capture time since boot: %s \n' % df.at[0,'date_time'])
                    debug_file.write('------capture start time since boot: %s \n' % str(start_time))
                    debug_file.write('------offset: %s \n' % str(df.at[0,'date_time']-start_time))
                    debug_file.write('------capture start epoch time: %s \n' % str(start_capture_epoch_time))
                    #debug_file.write('------capture start epoch time: %s \n' % str(df.at[0,'date_time']-start_time))
                    df['date_time'] = df['date_time'] - start_time + start_capture_epoch_time*1000000
                    #debug_file.write('------epoch in micros: %s \n' % str(df.at[0,'date_time'] - start_time + start_capture_time_micros))
                    debug_file.write('------val after adding offset: %s \n' % df.at[0,'date_time'])
                    df['date_time'] = pd.to_datetime(df['date_time'], unit='us')
                    df['date_time'] = df['date_time'].dt.tz_localize('UTC').dt.tz_convert('Asia/Jerusalem')
                    df['date_time'] = df['date_time'].dt.strftime('%H:%M:%S.%f')
                    debug_file.write('------%s saving file  \n' % str(time.time()))
                    debug_file.flush()

                    # Determine if server interface or client interface
                    if intf_index == server_intf_index:
                        ssrv_df = df
                    else:
                        clnt_df = df

                    if "5201" in conn_index:
                        csv_filename = os.path.join(res_dir, "%d_%s_%s.csv" % (int(time.time()), conn_index, intf_index))
                        df.to_csv(csv_filename)
                        debug_file.write('------%s file   saved\n' % str(time.time()))
                        debug_file.flush()

                # Create connection stat file out of df objects
                debug_file.write('-----%screating single connection DF \n' % str(time.time()))
                if ssrv_df is not None and clnt_df is not None:
                    conn_stat_obj = OnlineSingleConnStatistics(in_df=clnt_df, out_df=ssrv_df, interval_accuracy=3,
                                                               rtr_q_filename=q_disc_file)
                    csv_filename = os.path.join(res_dir, "%d_%s.csv" % (int(time.time()), conn_index))
                    conn_stat_obj.conn_df.to_csv(csv_filename)
            # Kill the clsact, so no more packets will be counted
            debug_file.write("End of capture: deleting BPF settings\n")
            debug_file.flush()

            debug_file.write('------%sbefore clear arrays\n' % str(time.time()))
            debug_file.flush()
            #pkt_array.items_delete_batch() !!! Not supported in ubuntu 18.04
            pkt_array.clear()
            debug_file.write('------%s cleared pkt_ array \n' % str(time.time()))
            debug_file.flush()
            #pkt_array_ext.items_delete_batch()!!! Not supported in ubuntu 18.04
            pkt_array_ext.clear()
            debug_file.write('------%s cleared pkt_ array ext\n' % str(time.time()))
            debug_file.flush()
            pkt_count_table.clear()
            pkt_out_count_table.clear()
            debug_file.write('cleared arrays\n')
            debug_file.flush()


    except KeyboardInterrupt:
        debug_file.write('KB Exception\n')
        debug_file.flush()
        pass

    except Exception as err:
        debug_file.write('Exception2\n')
        print(err, file=debug_file)
        traceback.print_exc(file=debug_file)
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print(repr(traceback.format_exception(exc_type, exc_value,
                                              exc_traceback)), file=debug_file)
        debug_file.flush()
    finally:
        debug_file.write('in finally\n')
        debug_file.flush()
        try:
            for intf_index in inft_index_list:
                # ingress
                ipr.tc("del", "clsact", intf_index)

        except Exception as err:
            debug_file.write('Exception3\n')
            print(err, file=debug_file)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print(repr(traceback.format_exception(exc_type, exc_value,
                                                  exc_traceback)), file=debug_file)
            debug_file.flush()
    pass


except Exception as err:
    print(err, file=debug_file)
