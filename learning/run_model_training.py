"""
if __name__ == 'main':
    for bg_flow in BG_FLOWS:
        learning.model_training.main(bg_flow=bg_flow)
"""
from learning import model_training
#from learning.env import absolute_path
import os
from learning.env import *
import subprocess
import sys
if __name__ == '__main__':
    # Automatic graphs generation:
    num_of_congestion_controls = 3
    num_of_time_samples = 10000
    unused_parameters = ['Capture Time Gap', 'In Throughput', 'Out Throughput', 'deepcci']
    for IS_DEEPCCI in [True]:
        bg_flows = [0]#, 15, 30, 75]
        for bg_flow in bg_flows:
            training_files_path = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/physical data/10 seconds/0 filter/3CC/no bottleneck'
            graphs_path = os.path.join(absolute_path,
                                       r'graphs/thesis_prime/physical/10 seconds/diverse chunk sizes/15 background flows/0 filter',
                                       str(bg_flow) + "_bg_flows_")
            num_of_classification_parameters = NUM_OF_CLASSIFICATION_PARAMETERS - len(unused_parameters)
            for chunk_size in [1000]:#, 3000, 6000, 10000, 30000, 60000]:
                DEEPCCI_NUM_OF_TIME_SAMPLES = int(chunk_size / 1000)  # 10
                model_training.main(training_files_path, bg_flow, IS_SAMPLE_RATE, IS_SAMPLE, IS_DEEPCCI, IS_FULLY_CONNECTED, num_of_classification_parameters,
                                    chunk_size, num_of_congestion_controls, num_of_time_samples, DEVICE, graphs_path, IS_TEST_ONLY, model_path, diverse_training_folder, IS_DIVERSE, DEEPCCI_NUM_OF_TIME_SAMPLES)
                #cmd = ['python', 'model_training.py']
                #theproc = subprocess.Popen([sys.executable, "model_training.py"]).wait()
                #os.system("script2.py 1")