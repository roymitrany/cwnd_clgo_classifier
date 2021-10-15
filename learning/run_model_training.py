"""
if __name__ == 'main':
    for bg_flow in BG_FLOWS:
        learning.model_training.main(bg_flow=bg_flow)
"""
from learning import model_training
from learning.env import absolute_path
import os
import subprocess
import sys
if __name__ == '__main__':
    # Automatic graphs generation:
    for IS_DEEPCCI in [False, True]:
        number_of_flows = [0, 15, 30, 75]
        for flow in number_of_flows:
            training_files_path = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/physical data/10 seconds/0 filter/3CC/no bottleneck'
            graphs_path = os.path.join(absolute_path,
                                       r'graphs/thesis_prime/physical/10 seconds/diverse chunk sizes/15 background flows/0 filter',
                                       str(flow) + "_bg_flows_")
            for CHUNK_SIZE in [1000, 3000, 6000, 10000, 30000, 60000]:
                model_training.main(training_files_path, flow)
                #cmd = ['python', 'model_training.py']
                #theproc = subprocess.Popen([sys.executable, "model_training.py"]).wait()
                #os.system("script2.py 1")