#!/usr/bin/python3
# Copyright (c) PLUMgrid, Inc.
# Licensed under the Apache License, Version 2.0 (the "License")
import shutil
from datetime import datetime
import os
from pathlib import Path
import socket

print("startinggg tcp_smart_dump, before inport")
import sys

# print(sys.path)

sys.path.insert(0, '..')
sys.path.insert(0, 'cwnd_clgo_classifier')
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
from online_sim.online_single_connection_statistics import OnlineSingleConnStatistics

# TODO: get interval accuracy as parameter
interval_accuracy = 3

# traffic_duration = int(sys.argv[1])
simulation_name = sys.argv[1]
total_duration = sys.argv[2]
srv_intf = sys.argv[3]
clnt_intf_list = sys.argv[4:]
inft_index_list = []
ipr = IPRoute()
interval_duration = int(total_duration)/10
# num_of_samples = int(traffic_duration / interval_duration)
server_intf_index = -1
res_root_dir = os.path.join(Path(os.getcwd()).parent, "raw_data")
debug_file_name = "./debug_files/%d_ebpf_debug.txt" % int(time.time())
debug_file = open(debug_file_name, 'w')
init_time = time.time()
print("-----%s starting tcp_smart_dump" % str(init_time), file=debug_file)
while True:
    if int(time.time() - init_time) > int(total_duration):
        # The script ran to the required duration, stop it.
        sys.exit()
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
        debug_file.write("jjjjjj\n")

        # handle server facing interface first, read egress traffic only
        server_intf_index = ipr.link_lookup(ifname=srv_intf)[0]
        inft_index_list.append(server_intf_index)
        print("server interface %s is: %d" % (srv_intf, server_intf_index), file=debug_file)
        try:
            ipr.tc("add", "clsact", server_intf_index)
        except Exception as err:
            debug_file.write("fail to add clsact to server interface\n")
            print(err, file=debug_file)

        debug_file.write("kkkkk\n")
        debug_file.flush()

        # egress
        # ipr.tc("add-filter", "bpf", server_intf_index, ":1", fd=fn.fd, name=fn.name,
        #       parent="ffff:fff3", classid=1, direct_action=True)
        # debug_file.write("Egress BPF Filter Added for interface intf!!!!\n")
        # debug_file.flush()
        ffilter = b.load_func("out_filter", BPF.SOCKET_FILTER)
        BPF.attach_raw_socket(ffilter, srv_intf)

        # frontend part
        socketfd = ffilter.sock

        sockobj = socket.fromfd(socketfd, socket.AF_PACKET, socket.SOCK_RAW, socket.IPPROTO_IP)
        sockobj.setblocking(True)

        # ingress
        for intf in clnt_intf_list:
            intf_index = ipr.link_lookup(ifname=intf)[0]
            inft_index_list.append(intf_index)
            print("if %s is: %d" % (intf, intf_index), file=debug_file)
            try:
                ipr.tc("add", "clsact", intf_index)
            except Exception as err:
                debug_file.write("fail to add clsact to server interface\n")
                print(err, file=debug_file)

            # ingress
            debug_file.write('adding  ingress filter for interface: %d\n' % intf_index)
            ipr.tc("add-filter", "bpf", intf_index, ":1", fd=fn.fd, name=fn.name,
                   parent="ffff:fff2", classid=1, direct_action=True)

        try:
            # Create results directory for this iteration
            tn = datetime.now()
            time_str = str(tn.month) + "." + str(tn.day) + "." + str(tn.year) + "@" + str(tn.hour) + "-" + str(
                tn.minute) + "-" + str(tn.second)
            res_dir = os.path.join(res_root_dir, time_str + '_' + simulation_name)
            os.mkdir(res_dir, 0o777)
            # Gather statistics from the router:
            # q_disc_file = os.path.join(res_dir, "%d_qdisc.csv" % int(time.time()))
            # q_disc_cmd = os.path.join(Path(os.getcwd()).parent, "simulation", "tc_qdisc_implementation.py")
            # command_line = 'python3 %s %s %s %d' % (q_disc_cmd, sys.argv[3], q_disc_file, interval_accuracy)
            # args = shlex.split(command_line)
            # q_proc = subprocess.Popen(args)
            start_capture_epoch_time = time.time()
            sniff_mode_table[0] = c_uint32(1)
            start_time_table[0] = c_uint64(0)  # reset start time
            time.sleep(interval_duration)
            sniff_mode_table[0] = c_uint32(0)
            # q_proc.send_signal(SIGINT)
            packets_count = pkt_count_table[0].value

            # If nothing was captured, delete the new directory and continue
            if packets_count == 0:
                shutil.rmtree(res_dir)
                continue
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
                    debug_file.write('-----%screating dataframe for %s\n' % (str(time.time()),intf_index))
                    debug_file.flush()
                    df = pd.DataFrame(df_dict[conn_index][intf_index],
                                      columns=['conn_index', 'date_time', 'length', 'ts_val', 'seq_num'])
                    df.index.name = "num"

                    # Get the capture start time, which is the number of microseconds from boot time until capture
                    # started.
                    # Then subtract the start time from packet capture time, which is the  number of microseconds from
                    # boot time until the packet was captured
                    # Add this result to epoch time in nanos at capture start time
                    # the result is the epoch time in micros when the packet was captured.
                    start_time = start_time_table[0].value
                    debug_file.write('------packet capture time since boot: %s \n' % df.at[0, 'date_time'])
                    debug_file.write('------capture start time since boot: %s \n' % str(start_time))
                    debug_file.write('------offset: %s \n' % str(df.at[0, 'date_time'] - start_time))
                    debug_file.write('------capture start epoch time: %s \n' % str(start_capture_epoch_time))
                    # debug_file.write('------capture start epoch time: %s \n' % str(df.at[0,'date_time']-start_time))
                    df['date_time'] = df['date_time'] - start_time + start_capture_epoch_time * 1000000
                    # debug_file.write('------epoch in micros: %s \n' % str(df.at[0,'date_time'] - start_time + start_capture_time_micros))
                    debug_file.write('------val after adding offset: %s \n' % df.at[0, 'date_time'])
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

                    csv_filename = os.path.join(res_dir,
                                                "%d_%s_%s.csv" % (int(time.time()), conn_index, intf_index))
                    df.to_csv(csv_filename)
                    debug_file.write('------%s file %s saved\n' % (str(time.time()), csv_filename))
                    debug_file.flush()

                # Create connection stat file out of df objects
                # For now, we analyze the connections offline!!
                '''if ssrv_df is not None and clnt_df is not None:
                    debug_file.write('-----%screating single connection DF \n' % str(time.time()))

                    conn_stat_obj = OnlineSingleConnStatistics(in_df=clnt_df, out_df=ssrv_df, interval_accuracy=3)
                    # rtr_q_filename=q_disc_file)
                    csv_filename = os.path.join(res_dir, "%d_%s.csv" % (int(time.time()), conn_index))
                    conn_stat_obj.conn_df.to_csv(csv_filename)
                else:
                    debug_file.write('-----%scould not create single connection DF \n' % str(time.time()))'''

            # Kill the clsact, so no more packets will be counted
            debug_file.write("End of capture: deleting BPF settings\n")
            debug_file.flush()

            debug_file.write('------%sbefore clear arrays\n' % str(time.time()))
            debug_file.flush()
            pkt_array.items_delete_batch()  # !!! Not supported in ubuntu 18.04
            # pkt_array.clear()
            debug_file.write('------%s cleared pkt_ array \n' % str(time.time()))
            debug_file.flush()
            pkt_array_ext.items_delete_batch()  # !!! Not supported in ubuntu 18.04
            # pkt_array_ext.clear()
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
