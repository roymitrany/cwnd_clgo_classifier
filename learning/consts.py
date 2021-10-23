from enum import Enum

class NetEnum(Enum):
    MY_NET = 1
    DEEPCCI_NET = 2
    FULLY_CONNECTED_NET = 3

D_10S_3CC_0F_NB_PATH = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/physical data/10 seconds/0 filter/3CC/no bottleneck'
D_10S_3CC_0F_B_PATH = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/physical data/10 seconds/0 filter/3CC/bottleneck'
D_60S_3CC_0F_0BG_B_PATH = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/physical data/60 seconds/0 filter/3CC/0BG/bottleneck'

R_10S_3CC_0F_NB_PATH =  r'/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/thesis_prime/physical/10 seconds/diverse chunk sizes'
CBIQ_UNUSED_PARAMETERS = ['Capture Time Gap', 'In Throughput', 'Out Throughput', 'deepcci']
DEEPCCI_UNUSED_PARAMETERS = ['timestamp', 'CBIQ', 'In Throughput', 'Out Throughput', 'Capture Time Gap']
THROUGHPUT_UNUSED_PARAMETERS = ['CBIQ', 'deepcci', 'Capture Time Gap']

SLEEP_DURATION = 60*60*0.1
