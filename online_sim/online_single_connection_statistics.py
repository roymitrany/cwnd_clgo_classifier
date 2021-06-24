import re
import sys
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import cycler, gridspec

from simulation.tcpdump_statistics import TcpdumpStatistics


def time_str_to_timedelta(time_str):
    my_time = datetime.strptime(time_str, "%H:%M:%S.%f")
    my_timedelta = my_time - datetime(1900, 1, 1)
    return my_timedelta


def get_delta(curr_timedelta, base_timedelta):
    """Get Seconds from time."""
    return (curr_timedelta - base_timedelta).seconds + (curr_timedelta - base_timedelta).microseconds / 1000000




class SingleConnStatistics:
    def __init__(self, interval_accuracy, rtr_q_filename):
        self.conn_df = self.in_conn_df = self.out_conn_df = None
        self.interval_accuracy = interval_accuracy
        self.rtr_q_filename = rtr_q_filename

    def build_df(self):
        in_passed_df, in_dropped_df, in_retransmit_df = self.reduce_dropped_packets(self.in_conn_df)

        # If we want accurate capture time, we need to do it before we round times by interval accuracy
        capture_time_column_name = "Capture Time Gap"
        self.calc_capture_delta_time(capture_time_column_name)

        # Round time by interval accuracy
        self.in_conn_df['date_time'] = self.in_conn_df['date_time'].map(
            lambda x: x[0:self.interval_accuracy - 6])
        self.out_conn_df['date_time'] = self.out_conn_df['date_time'].map(
            lambda x: x[0:self.interval_accuracy - 6])

        # Create a DF with all possible time ticks between min time to max time
        in_start_time = self.in_conn_df['date_time'].iloc[0]
        in_end_time = self.in_conn_df['date_time'].iloc[-1]
        out_start_time = self.out_conn_df['date_time'].iloc[0]
        out_end_time = self.out_conn_df['date_time'].iloc[-1]
        start_timedelta = min(in_start_time, out_start_time)
        end_timedelta = max(in_end_time, out_end_time)
        millies = 10 ** (3 - self.interval_accuracy)
        tdi = pd.timedelta_range(start_timedelta, end_timedelta, freq='%dL' % millies)
        self.conn_df = tdi.to_frame(name="Time")


        # Complete the capture delta time setup by getting the maximal value for each time slot
        capture_delta_df = pd.DataFrame(self.in_conn_df.groupby('date_time')[capture_time_column_name].max())
        self.conn_df = self.conn_df.join(capture_delta_df[capture_time_column_name])

        self.count_throughput(self.in_conn_df, 'In Throughput')
        self.count_throughput(self.out_conn_df, 'Out Throughput')
        self.count_throughput(in_passed_df, 'In Goodput')



        # Calculate CBIQ
        in_temp_df = self.create_seq_df(self.in_conn_df, 'in_seq_num')
        out_temp_df = self.create_seq_df(self.out_conn_df, 'out_seq_num')
        temp_df = in_temp_df.merge(out_temp_df, how='inner', on=['date_time'])
        temp_df = temp_df.set_index('date_time')
        temp_df['CBIQ'] = temp_df['in_seq_num'] - temp_df['out_seq_num']
        temp_df = temp_df.drop(columns=['in_seq_num', 'out_seq_num'])
        self.conn_df = self.conn_df.join(temp_df)

        self.count_dropped_packets(in_dropped_df, 'Connection Num of Drops')
        self.count_retransmit_packets(in_retransmit_df, 'Connection Num of Retransmits')
        self.count_ts_val(self.out_conn_df, "Send Time Gap")
        # ts_val_df = self.create_ts_val_df(in_conn_lines, self.interval_accuracy)

        self.conn_df.index.name = 'Time'

        # The gap between the total in and the total out indicates what's in the queue. We want to convert form
        # Mbps to Bytes

        if self.rtr_q_filename is not None:
            # Add qdisc columns, only for existing keys (inner join)
            qdisc_df = pd.read_csv(self.rtr_q_filename, sep="\t", header=None)
            qdisc_df.columns = ['Time', 'Total Bytes in Queue', 'Num of Packets', 'Num of Drops']
            reference_start_time = qdisc_df['Time'][0]
            qdisc_df['Time'] = qdisc_df['Time'].map(lambda time_str: time_str_to_timedelta(time_str))
            qdisc_df = qdisc_df.set_index('Time')
            self.conn_df = self.conn_df.join(qdisc_df, lsuffix='_caller')
            self.conn_df = self.conn_df.fillna(method='ffill')

        # Convert the time string into time offset float
        #self.conn_df['timestamp'] = self.conn_df['Time'].map(lambda x: get_delta(x, self.conn_df['Time'][0]))
        self.conn_df['timestamp'] = self.conn_df['Time'].map(lambda x: get_delta(x, reference_start_time))
        self.conn_df = self.conn_df.set_index('timestamp')
        self.conn_df = self.conn_df.drop(columns=['Time'])

        # Fill all Nan with 0 (we don't know anything better for what's left)
        self.conn_df = self.conn_df.fillna(0)

        return

    def calc_capture_delta_time(self, column):
        # calculate the packet arrival time difference
        # since we want to maintain the data based on time intervals, we should expect some intervals to include
        # a lot of packets, and others with only few of them.
        # It is hard to process such information, so we will
        # extract the maximal time gap between two sent packets for each interval.
        time_gap_series = pd.to_datetime(self.in_conn_df['date_time']).diff()
        time_gap_series = time_gap_series.convert_dtypes()
        time_gap_series = time_gap_series.fillna(0)
        time_gap_series = time_gap_series/1000000 #convert from nano to milli
        time_gap_series = time_gap_series.astype('int64')
        print('a')


        # Join it to th conn df
        time_gap_series.name = column
        self.in_conn_df = self.in_conn_df.join(time_gap_series)


    def count_ts_val(self, conn_df, column):
        # TS val indicates the timestamp in which the packet was sent.
        # since we want to maintain the data based on time intervals, we should expect some intervals to include
        # a lot of packets, and others with only few of them.
        # It is hard to process such information, so we will
        # extract the maximal time gap between two sent packets for each interval.

        # Create diff df first, with delta between ts_vals (this is the time gap
        time_gap_df = pd.DataFrame(conn_df['ts_val'].diff())
        time_gap_df.fillna(0)

        # Join it to th conn df
        time_gap_df = time_gap_df.rename(columns={"ts_val": column})
        conn_df = conn_df.join(time_gap_df)

        # Take the maximal value for each time slot
        bytes_per_timeslot_df = pd.DataFrame(conn_df.groupby('date_time')[column].max())
        self.conn_df = self.conn_df.join(bytes_per_timeslot_df[column])

    def count_throughput(self, conn_df, column):
        bytes_per_timeslot_df = pd.DataFrame(conn_df.groupby('date_time')['length'].sum())
        self.conn_df = self.conn_df.join(bytes_per_timeslot_df)
        self.conn_df = self.conn_df.rename(columns={"length": column})
        self.conn_df[column] = self.conn_df[column].map(lambda num: num * 8 / 10 ** (6 - self.interval_accuracy))
        values = {column: 0}
        self.conn_df = self.conn_df.fillna(value=values)

    def count_dropped_packets(self, dropped_df, column):
        drop_count_df = pd.DataFrame(dropped_df['date_time'].value_counts())
        self.conn_df = self.conn_df.join(drop_count_df)
        self.conn_df = self.conn_df.rename(columns={"date_time": column})
        values = {column: 0}
        self.conn_df = self.conn_df.fillna(value=values)

    def count_retransmit_packets(self, ret_df, column):
        ret_count_df = pd.DataFrame(ret_df['date_time'].value_counts())
        self.conn_df = self.conn_df.join(ret_count_df)
        self.conn_df = self.conn_df.rename(columns={"date_time": column})
        values = {column: 0}
        self.conn_df = self.conn_df.fillna(value=values)

    def create_seq_df(self, conn_df, col_name):
        temp_df = conn_df.drop_duplicates(subset=["date_time"],keep='first')
        temp_df = temp_df[['date_time', 'seq_num']]
        temp_df.columns = ['date_time', col_name]
        return temp_df

    @staticmethod
    def reduce_dropped_packets(conn_df):
        # The method assumes single connection. If lines are from multiple connections, two messages
        # from two different connections with the same seq number will be interpreted as retransmissions
        # If data was sent multiple times, the method keeps only the last retransmission.
        # The method assumes that add df records represent packets with data

        # There is a pd command to filter out duplicates, keeping the last instance
        passed_df = conn_df.drop_duplicates(subset='seq_num', keep='last')

        # Find the items that represent drops, and keep them in a separate df
        dups_series = conn_df.duplicated(subset='seq_num', keep='last')
        dups_only = dups_series[dups_series == True]
        fdf = pd.DataFrame(dups_only).join(conn_df)
        dropped_df = fdf.drop(columns=[0])

        # Find the items that represent retransmissions, and keep them in a separate df
        dups_series = conn_df.duplicated(subset='seq_num', keep='first')
        dups_only = dups_series[dups_series == True]
        fdf = pd.DataFrame(dups_only).join(conn_df)
        retransmit_df = fdf.drop(columns=[0])

        return passed_df, dropped_df, retransmit_df


class OnlineSingleConnStatistics(SingleConnStatistics):
    def __init__(self, in_df = None, out_df=None, in_file=None, out_file=None, interval_accuracy=3, rtr_q_filename=None):
        super().__init__(interval_accuracy, rtr_q_filename)

        # Either in/out DF or in/out files should be filled. We trus the code and don't check
        if in_file:
            self.in_conn_df = pd.read_csv(in_file, sep=",")
            self.out_conn_df = pd.read_csv(out_file, sep=",")
        else:
            self.in_conn_df = in_df
            self.out_conn_df = out_df

        self.in_conn_df = self.in_conn_df.sort_values(by=['date_time'])
        self.out_conn_df = self.out_conn_df.sort_values(by=['date_time'])
        self.build_df()


if __name__ == '__main__':
    intv_accuracy = 3
    abs_path = "/home/dean/PycharmProjects/cwnd_clgo_classifier/classification_data/online/6.20.2021@14-50-47_1_reno_2_bbr_3_cubic"
    in_file = abs_path + "/1624189855_167772170_64501_167773706_5201_8.csv"
    out_file = abs_path + "/1624189855_167772170_64501_167773706_5201_9.csv"
    rtr_file = abs_path + "/1624189847_qdisc.csv"
    q_line_obj = OnlineSingleConnStatistics(in_file=in_file, out_file=out_file, interval_accuracy= intv_accuracy, rtr_q_filename=rtr_file)
    q_line_obj.conn_df.to_csv(abs_path + '/single_connection_stat_debug.csv')
    # q_line_obj = OfflineSingleConnStatistics(in_file, out_file, rtr_file, intv_accuracy)
    # q_line_obj.conn_df.to_csv(abs_path + "/single_connection_stat_2.csv")