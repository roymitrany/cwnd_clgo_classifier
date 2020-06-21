import signal
import subprocess
from datetime import datetime
from subprocess import Popen
from sys import argv
import re  # xu
import time

run = True


def signal_handler(signal, frame):
    global run
    # print ("exiting")
    run = False


if __name__ == '__main__':
    queue_len_bytes_dict = {}
    # queue_len_packets_list = []
    assert argv != 3, "argv was %d. add ifname and filename as parameters" % len(argv)
    if_name = argv[1]
    results_filename = argv[2]
    signal.signal(signal.SIGINT, signal_handler)
    with open("qdisc_debug.txt", 'w') as qout:
        while run:
            p = Popen(["/sbin/tc", "-s", "qdisc", "show", "dev", if_name], stdout=subprocess.PIPE,
                      universal_newlines=True)
            output = p.communicate()
            qout.write(output[0] + "\n")
            match = re.search("backlog\s+(\d+[kK]?)b\s+(\d+)p", output[0])
            if match:

                if output[0].find("K") == -1:
                    num_of_bytes = match.group(1)
                else:
                    num = int(match.group(1)[0:-1]) * 1000
                    num_of_bytes = str(num)
                num_of_packets = match.group(2)
                queue_len_bytes_dict[datetime.now().strftime("%H:%M:%S.%f")[:-5]] = "%s\t%s" % (
                num_of_bytes, num_of_packets)
            time.sleep(0.1)
        # SIGINT was called. Save all results in a file:
    with open(results_filename, 'w') as outfile:
        for key, val in queue_len_bytes_dict.items():
            outfile.write("%s\t%s\n" % (key, val))
    qout.close()
    outfile.close()
