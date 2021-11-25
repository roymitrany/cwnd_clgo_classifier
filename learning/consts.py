import torch

# Debugging data:

DEBUG = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/physical data/10 seconds/bottleneck/DEBUG_1_DATAFRAME/STATS'

# Classification data:

ABSOLUTE_PATH = r'/home/dean/PycharmProjects/cwnd_clgo_classifier/'

D_60S_6CC_0F_B_R_PATH = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/physical data/60 seconds/0 filter/6 cc'
D_60S_60_6CC_0F_B_R_PATH = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/physical data/60 seconds/0 filter/bottleneck'



CBIQ_UNUSED_PARAMETERS = ("CBIQ", ['Capture Time Gap', 'In Throughput', 'Out Throughput', 'deepcci'])
CBIQ_WITHOUT_SEQ_NUMM_UNUSED_PARAMETERS = ("CBIQ", ['Capture Time Gap', 'In Throughput', 'Out Throughput', 'In Throughput', 'Out Throughput', 'deepcci'])
DEEPCCI_UNUSED_PARAMETERS = ("Deepcci", ['timestamp', 'CBIQ', 'In Throughput', 'Out Throughput', 'Capture Time Gap'])
IN_THROUGHPUT_UNUSED_PARAMETERS = ("In Throughput", ['timestamp', 'CBIQ',  'in_seq_num', 'out_seq_num', 'deepcci', 'Out Throughput', 'Capture Time Gap'])
OUT_THROUGHPUT_UNUSED_PARAMETERS = ("Out Throughput", ['timestamp', 'CBIQ',  'in_seq_num', 'out_seq_num','deepcci', 'In Throughput', 'Capture Time Gap'])
THROUGHPUT_UNUSED_PARAMETERS = ("Throughput", ['timestamp', 'CBIQ',  'in_seq_num', 'out_seq_num', 'deepcci', 'Capture Time Gap'])
CAPTURE_UNUSED_PARAMETERS = ("Capture Arrival Time", ['CBIQ',  'in_seq_num', 'out_seq_num', 'deepcci', 'In Throughput', 'Out Throughput'])
ALL_PARAMETERS_UNUSED_PARAMETERS = ("All Parameters", ['deepcci'])

SLEEP_DURATION = 60*60*1

NUM_OF_CLASSIFICATION_PARAMETERS = 8 # 6

IS_BATCH = True
IS_DIVERSE_DATA = False
IS_DATA_SAMPLE = False
IS_SAMPLE_RATE = True
SAVE_MODEL_PT = False
IS_FULL_SESSION = False

# GPU:
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Old classification data:

D_10S_3CC_0F_B_R_PATH = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/physical data/10 seconds/0 filter/3 cc/bottleneck/with retransmission'
D_10S_3CC_0F_B_NR_PATH = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/physical data/10 seconds/0 filter/3 cc/bottleneck/without retransmission'
D_10S_3CC_0F_NB_R_PATH = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/physical data/10 seconds/0 filter/3 cc/no bottleneck/with retransmission'
D_60S_3CC_0F_0BG_NB_PATH = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/physical data/60 seconds/0 filter/3CC/0BG/no bottleneck'
D_60S_3CC_0F_0BG_B_PATH = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/physical data/60 seconds/0 filter/3CC/0BG/bottleneck'
R_10S_3CC_0F_NB_PATH =  r'/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/thesis_prime/physical/10 seconds/diverse chunk sizes'
