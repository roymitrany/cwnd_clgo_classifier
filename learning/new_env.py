NUM_OF_CLASSIFICATION_PARAMETERS = 2 # timestemp & CBIQ
#NUM_OF_CLASSIFICATION_PARAMETERS = 1 # capture arrival time
#NUM_OF_CLASSIFICATION_PARAMETERS = 3 # timestemp & throughput
#NUM_OF_CLASSIFICATION_PARAMETERS = 5
NUM_OF_TIME_SAMPLES = 60000#9499# 10000 #60000# 9499 # 9499 # 60000
NUM_OF_HIDDEN_LAYERS = 100
CHUNK_SIZE = 30000#9499#10000#9499 # 9499
DEEPCCI_NUM_OF_TIME_SAMPLES = int(CHUNK_SIZE / 1000) # 10
NUM_OF_CONGESTION_CONTROL_LABELING = 3 # Reno, Cubic, & BBR
NUM_OF_CONV_FILTERS = 50
NUM_OF_EPOCHS = 100
NUM_OF_BATCHES = 10
BATCH_SIZE = 32
TRAINING_VALIDATION_RATIO = 0.3
START_AFTER = 0#6000
END_BEFORE = 0