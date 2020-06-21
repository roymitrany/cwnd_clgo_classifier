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
    drops_dict = {}
    # queue_len_packets_list = []
    assert argv != 3, "argv was %d. add ifname and filename as parameters" % len(argv)
    if_name = argv[1]
    results_filename = argv[2]
    signal.signal(signal.SIGINT, signal_handler)
    last_dropped = 0
    while run:
        p = Popen(["/sbin/tc", "-s", "qdisc", "show", "dev", if_name], stdout=subprocess.PIPE,
                  universal_newlines=True)
        output = p.communicate()

        # parse queue length
        match = re.search(r'dropped\s+(\d+).*backlog\s+(\d+[kK]?)b\s+(\d+)p', output[0], re.DOTALL)
        if match:
            drops_str = match.group(1)
            drops = int(drops_str) - last_dropped
            last_dropped = int(drops_str)

            if output[0].find("K") == -1:
                num_of_bytes = match.group(2)
            else:
                num = int(match.group(2)[0:-1]) * 1000
                num_of_bytes = str(num)
            num_of_packets = match.group(3)
            queue_len_bytes_dict[datetime.now().strftime("%H:%M:%S.%f")[:-5]] = "%s\t%s\t%d" % (
                num_of_bytes, num_of_packets, drops)


        time.sleep(0.1)
    # SIGINT was called. Save all results in a file:
    with open(results_filename, 'w') as outfile:
        for key, val in queue_len_bytes_dict.items():
            outfile.write("%s\t%s\n" % (key, val))
    outfile.close()
