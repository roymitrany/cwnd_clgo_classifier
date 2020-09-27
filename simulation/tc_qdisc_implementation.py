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
    output_file = open("q_disc_debug.txt", 'w')
    last_dropped = 0
    while run:
        p = Popen(["/sbin/tc", "-s", "qdisc", "show", "dev", if_name], stdout=subprocess.PIPE,
                  universal_newlines=True)
        output = p.communicate()
        output_file.write(output[0])
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
            time_str = datetime.now().strftime("%H:%M:%S.%f")[:-5]

            queue_len_bytes_dict[datetime.now().strftime("%H:%M:%S.%f")[:-5]] = "%s\t%s\t%d" % (
                num_of_bytes, num_of_packets, drops)

        time.sleep(0.093)

    # SIGINT was called. Save all results in a file:
    results_file = open(results_filename, 'w')
    for key, val in queue_len_bytes_dict.items():
        time_str = key
        num_of_bytes, num_of_packets, drops_str = val.split()
        results_file.write("%s\t%s\t%s\t%s\n" % (time_str, num_of_bytes, num_of_packets, drops_str))

    results_file.close()
    output_file.close()
