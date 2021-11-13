import torch

ABSOLUTE_PATH = r'/home/dean/PycharmProjects/cwnd_clgo_classifier/'

D_10S_3CC_0F_NB_PATH = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/physical data/10 seconds/0 filter/3CC/no bottleneck'
D_10S_3CC_0F_B_PATH = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/physical data/10 seconds/0 filter/3CC/bottleneck_cbiq'
D_60S_3CC_0F_0BG_NB_PATH = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/physical data/60 seconds/0 filter/3CC/0BG/no bottleneck'
D_60S_3CC_0F_0BG_B_PATH = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/physical data/60 seconds/0 filter/3CC/0BG/bottleneck'


R_10S_3CC_0F_NB_PATH =  r'/home/dean/PycharmProjects/cwnd_clgo_classifier/graphs/thesis_prime/physical/10 seconds/diverse chunk sizes'
CBIQ_UNUSED_PARAMETERS = ("CBIQ", ['Capture Time Gap', 'In Throughput', 'Out Throughput', 'deepcci'])
DEEPCCI_UNUSED_PARAMETERS = ("Deepcci", ['timestamp', 'CBIQ', 'In Throughput', 'Out Throughput', 'Capture Time Gap'])
IN_THROUGHPUT_UNUSED_PARAMETERS = ("In Throughput", ['CBIQ', 'deepcci', 'Out Throughput', 'Capture Time Gap'])
OUT_THROUGHPUT_UNUSED_PARAMETERS = ("Out Throughput", ['CBIQ', 'deepcci', 'In Throughput', 'Capture Time Gap'])
THROUGHPUT_UNUSED_PARAMETERS = ("Throughput", ['CBIQ', 'deepcci', 'Capture Time Gap'])
CAPTURE_UNUSED_PARAMETERS = ("Capture Arrival Time", ['CBIQ', 'deepcci', 'In Throughput', 'Out Throughput'])
ALL_PARAMETERS_UNUSED_PARAMETERS = ("All Parameters", ['Capture Time Gap', 'deepcci'])

SLEEP_DURATION = 60*60*0.1

NUM_OF_CLASSIFICATION_PARAMETERS = 6

IS_BATCH = True
IS_DIVERSE_DATA = False
IS_DATA_SAMPLE = False
IS_SAMPLE_RATE = True
SAVE_MODEL_PT = False

# GPU:
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
