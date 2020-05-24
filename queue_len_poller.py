import signal
import subprocess
from subprocess import Popen
from sys import argv
import re
import time

run = True

def signal_handler(signal, frame):
    global run
    #print ("exiting")
    run = False


if __name__ == '__main__':
    queue_len_bytes_list = []
    queue_len_packets_list = []
    assert argv != 3, "argv was %d. add ifname and filename as parameters" % len(argv)
    if_name = argv[1]
    results_filename = argv[2]
    signal.signal(signal.SIGINT, signal_handler)
    while run:

        p = Popen(["/sbin/tc", "-s", "qdisc", "show",  "dev", if_name], stdout=subprocess.PIPE, universal_newlines=True)
        output = p.communicate()
        #print (output[0])
        match = re.search("backlog\s+(\d+[kK]?)b\s+(\d+)p", output[0])
        if match:
            if output[0].find("K") == -1:
                queue_len_bytes_list.append (match.group(1))
            else:
                num = int(match.group(1)[0:-1])*1000
                queue_len_bytes_list.append(str(num))
            queue_len_packets_list.append(match.group(2))

        time.sleep(0.01)


    # SIGINT was called. Save al the results in a file
    with open(results_filename, 'w') as outfile:
        for num in queue_len_bytes_list:
            outfile.write(num + "\n")