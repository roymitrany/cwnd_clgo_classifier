import torch

# Debugging data:

DEBUG = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/physical data/10 seconds/bottleneck/DEBUG_1_DATAFRAME/STATS'

# Classification data:

ABSOLUTE_PATH = r'/home/dean/PycharmProjects/cwnd_clgo_classifier/'

D_60S_60_6CC_0F_B_R_PATH = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/physical data/60 seconds/0 filter/bottleneck'
D_60S_60_6CC_0F_NB_R_PATH = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/physical data/60 seconds/0 filter/no bottleneck'

D_6S_3CC_CLOUD_PATH = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/physical data/cloud'

D_60S_6CC_01F_B_PATH = '/remote_disk/physical data filter/60 seconds rtr02/10 filter'
D_60S_6CC_025F_B_PATH = '/remote_disk/physical data filter/60 seconds rtr02/25 filter'
D_60S_6CC_05F_B_PATH = '/remote_disk/physical data filter/60 seconds rtr02/50 filter'
D_60S_6CC_075F_B_PATH = '/remote_disk/physical data filter/60 seconds rtr02/75 filter'
D_60S_6CC_09F_B_PATH = '/remote_disk/physical data filter/60 seconds rtr02/90 filter'
D_60S_6CC_099F_B_PATH = '/remote_disk/physical data filter/60 seconds rtr02/99 filter'

D_10S_MININET_0BF = '/data_disk/classification_data/with_data_repetition/queue_size_500/tcp_noise/0_bbr_cubic_reno_background_flows'
D_10S_MININET_75BF = '/data_disk/classification_data/with_data_repetition/queue_size_500/tcp_noise/7 5_bbr_cubic_reno_background_flows'

R_60S_60_6CC_0F_B_R_PATH = "/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/thesis_prime/physical/60 seconds/bottleneck vs no bottleneck/bottleneck"
R_60S_60_6CC_0F_NB_R_PATH = "/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/thesis_prime/physical/60 seconds/bottleneck vs no bottleneck/no bottleneck"

R_6S_CLOUD_PATH = "/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/cloud"

R_60S_6CC_01F_B_PATH = "/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/section 4/60 seconds/filters/10 filter"
R_60S_6CC_025F_B_PATH = "/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/section 4/60 seconds/filters/25 filter"
R_60S_6CC_05F_B_PATH = "/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/section 4/60 seconds/filters/50 filter"
R_60S_6CC_075F_B_PATH = "/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/section 4/60 seconds/filters/75 filter"
R_60S_6CC_09F_B_PATH = "/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/section 4/60 seconds/filters/90 filter"
R_60S_6CC_099F_B_PATH = "/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/section 4/60 seconds/filters/99 filter"

CBIQ_UNUSED_PARAMETERS = ("CBIQ", ['Capture Time Gap', 'In Throughput', 'Out Throughput', 'deepcci'])
CBIQ_WITHOUT_SEQ_NUM_UNUSED_PARAMETERS = ("CBIQ", ['CBIQ', 'Capture Time Gap', 'in_seq_num', 'out_seq_num', 'In Throughput', 'Out Throughput', 'deepcci'])
DEEPCCI_UNUSED_PARAMETERS = ("Deepcci", ['timestamp', 'CBIQ',  'in_seq_num', 'out_seq_num', 'In Throughput', 'Out Throughput', 'Capture Time Gap'])
IN_THROUGHPUT_UNUSED_PARAMETERS = ("In Throughput", ['timestamp', 'CBIQ',  'in_seq_num', 'out_seq_num', 'deepcci', 'Out Throughput', 'Capture Time Gap'])
OUT_THROUGHPUT_UNUSED_PARAMETERS = ("Out Throughput", ['timestamp', 'CBIQ',  'in_seq_num', 'out_seq_num','deepcci', 'In Throughput', 'Capture Time Gap'])
THROUGHPUT_UNUSED_PARAMETERS = ("Throughput", ['timestamp', 'CBIQ',  'in_seq_num', 'out_seq_num', 'deepcci', 'Capture Time Gap'])
CAPTURE_UNUSED_PARAMETERS = ("Capture Arrival Time", ['timestamp', 'CBIQ',  'in_seq_num', 'out_seq_num', 'deepcci', 'In Throughput', 'Out Throughput'])
ALL_PARAMETERS_UNUSED_PARAMETERS = ("All Parameters", ['deepcci'])

DRAGONFLY_UNUSED_PARAMETERS = ("Dragonfly", ['Capture Time Gap', 'deepcci'])


MININET_UNUSED_PARAMETERS = ["timestamp", "In Throughput", "Out Throughput", "Connection Num of Drops", "Connection Num of Retransmits", "Send Time Gap", "CBIQ", "Total Bytes in Queue", "Num of Packets", "Num of Drops"]
MININET_CBIQ_UNUSED_PARAMETERS = ("CBIQ", ["In Throughput", "Out Throughput", "Connection Num of Drops", "Connection Num of Retransmits", "Send Time Gap", "Total Bytes in Queue", "Num of Packets", "Num of Drops"])
MININET_THROUGHPUT_UNUSED_PARAMETERS = ("Throughput", ["timestamp", "CBIQ", "Out Throughput", "Connection Num of Drops", "Connection Num of Retransmits", "Send Time Gap", "Total Bytes in Queue", "Num of Packets", "Num of Drops"])
MININET_DRAGONFLY_UNUSED_PARAMETERS = ("Dragonfly", ["Out Throughput", "Connection Num of Drops", "Connection Num of Retransmits", "Send Time Gap", "Total Bytes in Queue", "Num of Packets", "Num of Drops"])
MININET_DEEPCCI_UNUSED_PARAMETERS = ("Deepcci", ["CBIQ", "In Throughput", "Out Throughput", "Connection Num of Drops", "Connection Num of Retransmits", "Send Time Gap", "Total Bytes in Queue", "Num of Packets", "Num of Drops"])

SLEEP_DURATION = 60*60*1

NUM_OF_CLASSIFICATION_PARAMETERS = 8 # 6
# NUM_OF_CLASSIFICATION_PARAMETERS = 10 # 6

IS_BATCH = True
IS_DIVERSE_DATA = False
IS_DATA_SAMPLE = False
IS_SAMPLE_RATE = True
SAVE_MODEL_PT = False
IS_FULL_SESSION = False
IS_MININET = False

# GPU:
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Old classification data:

D_10S_3CC_0F_B_R_PATH = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/physical data/10 seconds/0 filter/3 cc/bottleneck/with retransmission'
D_10S_3CC_0F_B_NR_PATH = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/physical data/10 seconds/0 filter/3 cc/bottleneck/without retransmission'
D_10S_3CC_0F_NB_R_PATH = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/physical data/10 seconds/0 filter/3 cc/no bottleneck/with retransmission'
D_60S_3CC_0F_0BG_NB_PATH = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/physical data/60 seconds/0 filter/3CC/0BG/no bottleneck'
D_60S_3CC_0F_0BG_B_PATH = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/physical data/60 seconds/0 filter/old/3CC/0BG/bottleneck'
R_10S_3CC_0F_NB_PATH =  r'/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/thesis_prime/physical/10 seconds/diverse chunk sizes'
