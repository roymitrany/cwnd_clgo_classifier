#!/home/another/PycharmProjects/cwnd_clgo_classifier/venv/bin/python3
import os
import subprocess
import sys
import asyncio

# MAX_PROCESSES = 2 # 14
MAX_PROCESSES = 14 # 14

#raw_data_path = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/asymetric traffic/raw_data'
#raw_data_path = '/data_disk/physical data/raw_data'
#raw_data_path = '/remote_disk/raw_data/physical/60 seconds/0_bg_flows'#/discrete_data'
#raw_data_path = '/remote_disk/raw_data/physical/6 congestion controls/10_sec_6_algos/rtr01'
raw_data_path = '/remote_disk/raw_data/physical/3 congestion controls/10 seconds/discrete_bg/sim14'
# raw_data_path = '/remote_disk/raw_data/physical/6 congestion controls/60_sec_6_algos/rtr02'
#raw_data_path = '/remote_disk/raw_data/physical/3 congestion controls/10 seconds no bottleneck'
#raw_data_path = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/physical data/10 seconds/bottleneck/DEBUG_1_DATAFRAME/RAW'
raw_data_path = '/remote_disk/raw_data/physical/6 congestion controls/60_sec_6_algos/rtr02'
raw_data_path = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/cloud/raw_data'
raw_data_path = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/cloud/60sec'


#raw_data_path = '/remote_disk/raw_data/physical/3 congestion controls/60 seconds'
#raw_data_path = '/remote_disk/raw_data/physical/3 congestion controls/60 seconds'
#raw_data_path = '/remote_disk/raw_data/physical/3 congestion controls/60 seconds no bottlneck 0 background flows'
#raw_data_path = '/remote_disk/raw_data/physical/3 congestion controls/60 seconds no bottlneck 0 background flows'
#raw_data_path = '/remote_disk/raw_data/physical/6 congestion controls/10_sec_6_algos/rtr02'
#raw_data_path = '/remote_disk/raw_data/new_topo/10_sec_6_algos/rtr01'
#dst_path = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/asymetric traffic/0.9 retransmission'
#dst_path = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/physical data/filtered_data_0.75'
#dst_path = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/physical_data/10 seconds/no bottleneck/filtered_data_0'
# dst_path = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/physical_data/60 seconds/filtered_data_0'
#dst_path = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/new_topo/10_sec_6_algos/rtr01'
#dst_path = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/physical data/60 seconds/0 filter/3CC/0BG/no bottleneck'
#dst_path = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/physical data/60 seconds/0 filter/6CC/no bottleneck'
#dst_path = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/physical data/60 seconds/0 filter/3CC/diverseBG/bottleneck'
dst_path = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/physical data/10 seconds/0 filter/3 cc/no bottleneck/with retransmission'
dst_path = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/physical data/60 seconds/0 filter/no bottleneck'
dst_path = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/physical data/60 seconds/0.5 filter'
dst_path = "/remote_disk/physical data/60 seconds/0.9 filter"
dst_path = "/remote_disk/physical data filter/60 seconds rtr02/0.99 filter"
dst_path = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/physical data/cloud'
dst_path = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/physical data/cloud_60_sec'
#dst_path = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/physical data/10 seconds/bottleneck/DEBUG_1_DATAFRAME/STATS'
#dst_path = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/physical data/60 seconds/0 filter/6CC/no bottleneck'

# abs_path = '/data_disk/physical_res'
# abs_path = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/no_tso_0_75_bg_flows_bw_max_100'
folders_list = os.listdir(raw_data_path)


# folders_list = ['8.12.2021@14-18-35_NumBG_0_LinkBW_1000_Queue_900']
# for folder in folders_list:
#    subprocess.Popen(['./online_filtering.py', abs_path, folder])

async def process_folder(folder, sem):
    async with sem:  # controls/allows running 10 concurrent subprocesses at a time
        proc = await asyncio.create_subprocess_exec(sys.executable, 'online_filtering.py', raw_data_path, dst_path,
                                                    folder)
        await proc.wait()


async def main():
    sem = asyncio.Semaphore(MAX_PROCESSES)
    await asyncio.gather(*[process_folder(folder, sem) for folder in folders_list])


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
loop.close()
