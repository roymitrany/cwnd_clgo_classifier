Dragonfly’s system README

Run Classification:
Run “run_model_training.py”- one can control the number of background flows, the sub- interval size, and the amount of dropped- sampling- as they are all adjustable parameters.
The paths to the data path and the results path (to be set as desired) are defined in consts.py
(see an example of such dataframe data attached in “learning/data”. We had 1500 of sessions such this one). 
Notice that the results directory is created automatically.

The implementation of the Dragonfly classification module can be found in “my_net.py”.
The implementation of the Deepcci classification module can be found in “deepcci_net.py”.

Create plots:
After classification- one can analyze the results with suitable graphs. This is implemented in “thesis_graphs_utils.py”.
To create such plots one should first choose the parameter with which he wishes to compare the classification results between Dragonfly and Deepcci
(i.e. background flows, sub- interval sizes, etc.) and then activate the function which creates the appropriate graph. 
Note that it is necessary to set the results path accordingly.

Run simulation (physical implementation- including cloud utilization):
To create more experiments- for training and for testing one must first implement the physical environment as described in the paper. 
The implementation of the physical simulation to be found within the “physical_sim” sub- directory.
After acquiring raw data- one needs to translate the data files to the system’s dataframes- using “online_filtering_launcher.py” which uses “online filtering.py”- 
this is where we calculate the inputs (i.e. CBIQ, etc.) and uses dropping samples- to be set as desired.

Run simulation (mininet implementation):
The implementation of the mininet emulation is to be found within the “simulation” sub- directory. To create dataframes for classification-
one needs to activate the “simulation_implemantion.py” file and choose the settings of the experiments.

The data is analyzed in “utills.py” and in “results_manager.py” for classification purposes.
