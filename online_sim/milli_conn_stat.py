import glob
import os
import re
import shutil
import sys

# print(sys.path)
from pathlib import Path
from time import sleep
from abc import ABC, abstractmethod


sys.path.append('/home/another/PycharmProjects/cwnd_clgo_classifier')
import pandas as pd
from datetime import datetime

#patch
cnt = 0

def time_str_to_timedelta(time_str):
    my_time = datetime.strptime(time_str, "%H:%M:%S.%f")
    my_timedelta = my_time - datetime(1900, 1, 1)
    return my_timedelta


def get_delta(curr_timedelta, base_timedelta):
    """Get Seconds from time."""
    return (curr_timedelta - base_timedelta).seconds + (curr_timedelta - base_timedelta).microseconds / 1000000


class SampleConnStat(ABC):
    def __init__(self, in_file=None, out_file=None, interval_accuracy=3):
        self.conn_df = None
        self.interval_accuracy = interval_accuracy

        self.in_conn_df = self.create_sample_df(in_file, None)
        self.out_conn_df = self.create_sample_df(out_file, self.in_conn_df)

        # If one of the DFs was not created, abort the procedure
        if self.in_conn_df is None or self.out_conn_df is None:
            raise ValueError("DF not created")

        if self.method is None:
            self.method = 'abstract'
        self.build_df()

    def build_df(self):
        self.in_conn_df = self.in_conn_df.sort_values(by=['date_time'])
        self.out_conn_df = self.out_conn_df.sort_values(by=['date_time'])

        # calculate the packet arrival time difference
        # since we want to maintain the data based on time intervals, we should expect some intervals to include
        # a lot of packets, and others with only few of them.
        # It is hard to process such information, so we will
        # extract the maximal time gap between two sent packets for each interval.
        time_gap_series = pd.to_datetime(self.in_conn_df['date_time']).diff()
        time_gap_series = time_gap_series.convert_dtypes()
        time_gap_series = time_gap_series.fillna(0)
        time_gap_series = time_gap_series / 1000  # convert from nano to micro
        time_gap_series = time_gap_series.astype('int64')

        # Join it to the conn df
        time_gap_series.name = "Capture Time Gap"
        self.in_conn_df = self.in_conn_df.join(time_gap_series)

        # Round time by interval accuracy
        self.in_conn_df['date_time'] = self.in_conn_df['date_time'].map(
            lambda x: x[0:self.interval_accuracy - 6])
        self.out_conn_df['date_time'] = self.out_conn_df['date_time'].map(
            lambda x: x[0:self.interval_accuracy - 6])

        # Create a DF with all possible time ticks, starting from min time for n periods
        in_start_time = self.in_conn_df['date_time'].iloc[0]
        out_start_time = self.out_conn_df['date_time'].iloc[0]
        start_timedelta = min(in_start_time, out_start_time)
        millies = 10 ** (3 - self.interval_accuracy)
        tdi = pd.timedelta_range(start_timedelta, periods=10000, freq='%dL' % millies)
        self.conn_df = tdi.to_frame(name="Time")

        # count inbound throughput and attach it to main conn df
        self.in_conn_df = self.calculate_throughput(self.in_conn_df, 'In Throughput')
        self.in_conn_df['Time'] = self.in_conn_df['date_time'].map(lambda time_str: time_str_to_timedelta(time_str))
        temp_df = self.in_conn_df[['Time', 'In Throughput', time_gap_series.name]]
        self.conn_df = pd.merge(self.conn_df, temp_df, on='Time', how='left')

        # count outbound throughput and attach it to main conn df
        self.out_conn_df = self.calculate_throughput(self.out_conn_df, 'Out Throughput')
        self.out_conn_df['Time'] = self.out_conn_df['date_time'].map(lambda time_str: time_str_to_timedelta(time_str))
        temp_df = self.out_conn_df[['Time', 'Out Throughput']]
        self.conn_df = pd.merge(self.conn_df, temp_df, on='Time', how='left')

        # Fill 0 when no throughput was recorded
        values = {'In Throughput': 0, 'Out Throughput': 0, time_gap_series.name: -1}
        self.conn_df = self.conn_df.fillna(value=values)

        # Calculate CBIQ
        self.calculate_cbiq()

        # Deepcci patch
        in_deepcci_df = self.in_conn_df[['Time', 'deepcci']]
        self.conn_df = pd.merge(self.conn_df, in_deepcci_df, on='Time', how='left')
        self.conn_df[['deepcci']] = self.conn_df[['deepcci']].fillna(0)
        # End patch

        self.conn_df.index.name = 'Time'


        # Convert the time string into time offset float
        self.conn_df['timestamp'] = self.conn_df['Time'].map(lambda x: get_delta(x, self.conn_df['Time'][0]))
        self.conn_df = self.conn_df.set_index('timestamp')
        self.conn_df = self.conn_df.drop(columns=['Time'])

        # Reorder column to make the DF similar to original single connection stat
        self.conn_df = self.conn_df[[time_gap_series.name, 'In Throughput', 'Out Throughput', 'CBIQ', 'deepcci']]
        return

    def calculate_cbiq(self):
        # Add in sequence column to conn DF, interpolate missing fields
        in_temp_df = self.create_seq_df(self.in_conn_df, 'in_seq_num')
        global cnt
        #in_temp_df.to_csv(os.path.join(abs_path, folder, "in_seq_%d_%s.csv"%(cnt, self.method)))
        #
        self.conn_df = pd.merge(self.conn_df, in_temp_df, on='Time', how='left')
        #self.conn_df = self.conn_df.fillna(method='ffill')
        self.conn_df = self.conn_df.interpolate()

        # Add out sequence column to conn DF, DO NOT interpolate missing fields
        out_temp_df = self.create_seq_df(self.out_conn_df, 'out_seq_num')
        #out_temp_df.to_csv(os.path.join(abs_path, folder, "out_seq_%d_%s.csv"%(cnt, self.method)))
        cnt+=1
        self.conn_df = pd.merge(self.conn_df, out_temp_df, on='Time', how='left')
        #self.conn_df = self.conn_df.fillna(method='ffill')
        self.conn_df = self.conn_df.interpolate()

        # clacullate CBIQ. Since out seq is not filled, only timeticks that have out seq
        # Will be calculated
        self.conn_df['CBIQ'] = self.conn_df['in_seq_num'] - self.conn_df['out_seq_num']
        self.conn_df = self.conn_df.drop(columns=['in_seq_num', 'out_seq_num'])
        self.conn_df = self.conn_df.fillna(method='ffill')
        #self.conn_df = self.conn_df.interpolate()

        # Convert to integer (interpolation created float values)
        self.conn_df['CBIQ'] = self.conn_df['CBIQ'].map(lambda x: int(x))



    # Create a df with one sequence number for each timeslot. If a DF contains more than one
    # Row for a timeslot, use the first one. Set timedelta as index
    def create_seq_df(self, conn_df, col_name):

        seq_df = conn_df.drop_duplicates(subset=["date_time"], keep='first')
        seq_df = seq_df[['date_time', 'seq_num']]
        seq_df.columns = ['date_time', col_name]
        seq_df['Time'] = seq_df['date_time'].map(lambda time_str: time_str_to_timedelta(time_str))
        seq_df = seq_df.drop(columns=['date_time'])
        return seq_df

    def create_sample_df(self, filename, ref_df=None):
        df = pd.read_csv(filename, sep=",")
        df = df.sort_values(by=['date_time'])

        # If there are less than 200 packets, traffic is too low, return with nothing
        if len(df.index) < 200:
            return
        df = remove_retransmissions(df)
        df['trunk'] = df['date_time'].str[:-3]

        # Temporary patch: count deepcci
        deepcci_df = pd.DataFrame(df.groupby('trunk')['length'].count())
        deepcci_df = deepcci_df.rename(columns={"length": 'deepcci'})
        deepcci_df.rename_axis('trunk')
        df = pd.merge(df, deepcci_df, on='trunk', how='left')

        # Take out all the packets that are not part of the sample, according to sample method in subclass
        df = self.reduce_packets(df, ref_df)
        df = df.drop(columns=['trunk'])
        return df

    @abstractmethod
    def calculate_throughput(self, conn_df, column):
        pass

    @abstractmethod
    def reduce_packets(self, df, ref_df):
        pass

class RandomSampleConnStat(SampleConnStat):
    def __init__(self, in_file=None, out_file=None, interval_accuracy=3, prob=.1):
        self.prob = prob
        self.method = 'random'
        super(RandomSampleConnStat, self).__init__(in_file, out_file, interval_accuracy)

    def calculate_throughput(self, conn_df, column):

        # Count all the bytes that arrive at any msec
        bytes_per_timeslot_df = pd.DataFrame(conn_df.groupby('date_time')["length"].sum())
        bytes_per_timeslot_df = bytes_per_timeslot_df.rename(columns={"length": column})

        # Merge with conn_df
        temp_df = pd.merge(conn_df, bytes_per_timeslot_df, on='date_time', how='left')        # Translate from Bytes per time tick to Mbps
        # Throughput calculation: from Bytes to bits, divided by accuracy (1000 for msec) and by probability
        temp_df[column] = temp_df[column].map(lambda num: num * 8 / (10 ** (6 - self.interval_accuracy)*(self.prob)))
        values = {column: 0}
        temp_df = temp_df.fillna(value=values)

        # The returned value should include at most one row per msec
        temp_df = temp_df.drop_duplicates(subset=["date_time"],keep='first')
        return temp_df


    def reduce_packets(self, df, ref_df):
        # Randomly drop 90% of the packets
        df = df.drop(df.sample(frac=1-self.prob).index)
        return df

class SelectiveSampleConnStat(SampleConnStat):
    def __init__(self, in_file=None, out_file=None, interval_accuracy=3, prob=.1):
        self.prob = prob
        self.method = 'selective'
        self.in_reduced_df = None
        self.out_reduced_df = None
        super(SelectiveSampleConnStat, self).__init__(in_file, out_file, interval_accuracy)

    def calculate_throughput(self, conn_df, column):

        # Count all the bytes that arrive at any msec
        bytes_per_timeslot_df = pd.DataFrame(conn_df.groupby('date_time')["length"].sum())
        bytes_per_timeslot_df = bytes_per_timeslot_df.rename(columns={"length": column})

        # Merge with conn_df
        temp_df = pd.merge(conn_df, bytes_per_timeslot_df, on='date_time', how='left')        # Translate from Bytes per time tick to Mbps
        # Throughput calculation: from Bytes to bits, divided by accuracy (1000 for msec) and by probability
        temp_df[column] = temp_df[column].map(lambda num: num * 8 / (10 ** (6 - self.interval_accuracy)*(self.prob)))
        values = {column: 0}
        temp_df = temp_df.fillna(value=values)

        # The returned value should include at most one row per msec
        temp_df = temp_df.drop_duplicates(subset=["date_time"],keep='first')
        return temp_df

    # override superclass function
    # use
    def calculate_cbiq(self):
        if self.in_reduced_df is not None and self.out_reduced_df is not None:
            seq_df = self.in_reduced_df[['date_time', 'seq_num']].join(self.out_reduced_df[['seq_num']],rsuffix='_out', lsuffix = '_in')
            seq_df['CBIQ'] = seq_df['seq_num_in']-seq_df['seq_num_out']
            seq_df['trunk'] = seq_df['date_time'].str[:-3]
            seq_df = seq_df.drop(columns=['seq_num_in', 'seq_num_out', 'date_time'])
            seq_df['Time'] = seq_df['trunk'].map(lambda time_str: time_str_to_timedelta(time_str))
            self.conn_df = self.conn_df.merge(seq_df, on='Time', how='left')
            self.conn_df = self.conn_df.drop(columns=['trunk'])
            self.conn_df = self.conn_df.fillna(method='ffill')
            print('1')
        else:
            self.conn_df['CBIQ'] = 0
            print('2')


    def reduce_packets(self, df, ref_df):
        df_list = []
        if ref_df is None:
            # Randomly drop 90% of the packets
            ret_df = df.drop(df.sample(frac=1-self.prob).index)
        else:
            # for each line in the reference DF, add the closest line in our df
            for i, row in ref_df.iterrows():
                # Create a series of date_time that are bigger than the date_time in the reference
                my_series = df[df.date_time>row['date_time']]['date_time']
                # Take the smallest value from the series
                my_val = my_series.min()
                # Find the index of this value in the DF, and add it to the list of inices to keep
                # (this command actually creates a one line df, and appends it to a list of one line DFs
                df_list.append(df[df.date_time==my_val])
            # Create a big DF og all the one line DFs
            ret_df = pd.concat(df_list)
            self.in_reduced_df = ref_df.reset_index()
            self.out_reduced_df = ret_df.reset_index()
            #df = df.drop(df.sample(frac=1-self.prob).index)
        return ret_df



class MilliSampleConnStat(SampleConnStat):
    def __init__(self, in_file=None, out_file=None, interval_accuracy=3):
        self.method = 'milli'

        super(MilliSampleConnStat, self).__init__(in_file, out_file, interval_accuracy)

    def calculate_throughput(self, conn_df, column):
        # bytes_per_timeslot_series = conn_df['seq_num'].diff()
        # bytes_per_timeslot_series = bytes_per_timeslot_series.fillna(0)
        # bytes_per_timeslot_series = bytes_per_timeslot_series.astype('int64')

        # bytes_per_timeslot_series.name = column
        temp_df = conn_df[['seq_num']]
        while True:
            bytes_per_timeslot_series = temp_df['seq_num'].diff()
            bytes_per_timeslot_series = bytes_per_timeslot_series.fillna(1)
            bytes_per_timeslot_series = bytes_per_timeslot_series.astype('int64')
            bytes_per_timeslot_series.name = column
            temp_df = temp_df.join(bytes_per_timeslot_series)

            temp_df1 = temp_df[(temp_df[[column]] >= 0).all(1)]
            if len(temp_df) == len(temp_df1):
                break  # no more out of order sequences
            # Get rid of all sequences with negative diff, that indicate out of order. Then repeat the loop
            # to locate more ooo seq.
            temp_df = temp_df1.drop(columns=[column])

        conn_df = conn_df.join(temp_df[column])
        conn_df = conn_df.fillna(0)
        # Translate from Bytes per time tick to Mbps
        conn_df[column] = conn_df[column].map(lambda num: num * 8 / 10 ** (6 - self.interval_accuracy))

        # Join it to the conn df
        # bytes_per_timeslot_series.name = column
        # conn_df = conn_df.join(bytes_per_timeslot_series)
        return conn_df

    def reduce_packets(self, df, ref_df):
        # Keep only the first packet for each timeslot
        df = df.drop_duplicates(keep='first', subset=['trunk'])
        return df


def remove_retransmissions(conn_df):
    # Remove retransmissions
    while True:
        t_series = conn_df['seq_num'].diff()
        if t_series.min() >= 0:
            break
        conn_df = conn_df.join(t_series, rsuffix='_diff')
        conn_df = conn_df.drop(conn_df[conn_df['seq_num_diff'] < 0].index)
        conn_df = conn_df.drop(columns=['seq_num_diff'])
    return conn_df





if __name__ == '__main__':
    #sleep(60*60*10)
    intv_accuracy = 3
    algo_list = ['reno', 'bbr',
                 'cubic']  # Should be in line with measured_dict keys in online_simulation.py main function
    #abs_path = "/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/online_classification/60_background_flows    
    abs_path = '/home/another/PycharmProjects/cwnd_clgo_classifier/classification_data/no_tso_0_75_bg_flows'
    #folders_list = os.listdir(abs_path)
    folders_list = ['8.11.2021@3-59-19_NumBG_10_LinkBW_300_Queue_600']
    for folder in folders_list:
        # Look for all raw results in the folder
        result_files = glob.glob(os.path.join(abs_path, folder, "*_6450[0-9]_*"))
        # If there are not enough csv files in the folder, this folder is not interesting and should be removed
        if len(result_files) < 4:
            print("not enough csv files in %s, deleting folder" % folder)
            shutil.rmtree(os.path.join(abs_path, folder))

        # if there are already sample files in the folder, continue
        '''result_files1 = glob.glob(os.path.join(abs_path, folder, "%s_sample*"%method))
        if len(result_files1) > 0:
            print("folder %s already analyzed" % folder)
            continue'''
        # find the destination interface
        if_dict = {}
        for res_file in result_files:
            search_obj = re.search(r'_(\d+).csv$', str(res_file))
            if search_obj:
                if_num = search_obj.group(1)
                if if_num in if_dict.keys():
                    if_dict[if_num] += 1
                else:
                    if_dict[if_num] = 1
        # The first interface with more than one in the value is the server interface
        server_if = 0
        for key, val in if_dict.items():
            if val > 1:
                server_if = int(key)

        # If no interface was spotted more than once, end the iteration
        if server_if == 0:
            continue

        # Find the client files, and make sure there is server file for it.
        # If there is, run stat on them
        for res_file in result_files:
            search_obj = re.search(r'\d+_(\d+_64\d+_\d+_\d+)_(\d+).csv', str(res_file))
            monitored_if = int(search_obj.group(2))
            if search_obj:
                if monitored_if == server_if:
                    continue

            # Look for a matching server file
            search_pattern = search_obj.group(1) + '_' + str(server_if)
            for res_file2 in result_files:
                if search_pattern in res_file2:
                    print("analyzing %s and %s" % (res_file, res_file2))
                    in_file = os.path.join(abs_path, folder, res_file)
                    out_file = os.path.join(abs_path, folder, res_file2)

                    # Create random and milli sample conn stats.
                    # If creation of one of the fails, do not continue for this connection
                    try:
                        #selective_scs = SelectiveSampleConnStat(in_file=in_file, out_file=out_file, interval_accuracy=intv_accuracy)
                        random_scs = RandomSampleConnStat(in_file=in_file, out_file=out_file, interval_accuracy=intv_accuracy)
                        #milli_scs = MilliSampleConnStat(in_file=in_file, out_file=out_file,
                        #                             interval_accuracy=intv_accuracy)
                        #scs_list = [selective_scs, random_scs, milli_scs]
                        scs_list = [random_scs]
                    except ValueError:
                        break # break the inner loop, continue the outer loop to the next connection in the folder

                    # Determine the cwnd algo from the source port
                    search_obj = re.search(r'[0-9]+_[0-9]+_6450([0-9])_[0-9]+_52[0-9][0-9]', str(res_file))
                    if not search_obj:
                        break
                    algo_id = int(search_obj.group(1)) - 1
                    algo_name = algo_list[algo_id]

                    for scs in scs_list:
                        sample_csv_file_name = '%s_sample_stat_%s_%d.csv' % (scs.method, algo_name, monitored_if)
                        scs.conn_df.to_csv(os.path.join(abs_path, folder, sample_csv_file_name))
