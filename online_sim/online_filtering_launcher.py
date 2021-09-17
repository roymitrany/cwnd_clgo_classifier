#!/home/another/PycharmProjects/cwnd_clgo_classifier/venv/bin/python3
import os
import subprocess
import sys
import asyncio

MAX_PROCESSES = 14

raw_data_path = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/sim19/raw_data'
#raw_data_path = '/data_disk/physical data/diverse_bg/raw_data'
dst_path = '/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/sim19/filtered_data'
#dst_path = '/data_disk/physical data/diverse_bg/filtered_data_0.9_filter'
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
