import pandas as pd
import numpy as np
import copy
import random
import itertools
from operator import itemgetter
import pm4py
import getopt, sys, os


def _sort_log(log):
    log = copy.deepcopy(log)
    log = sorted(log.to_dict('records'), key=lambda x: x['case_id'])
    for key, group in itertools.groupby(log, key=lambda x: x['case_id']):
        events = list(group)
        events = sorted(events, key=itemgetter('end_timestamp'))
        length = len(events)
        for i in range(0, len(events)):
            events[i]['pos_trace'] = i + 1
            events[i]['trace_len'] = length
    log = pd.DataFrame.from_dict(log)
    log.sort_values(by='end_timestamp', inplace=True)
    return log

def trainTestSplit(df, test_len=0.3, one_timestamp=False):
    log = df
    log = _sort_log(log)
    # log = self.log.data.to_dict('records')
    num_events = int(np.round(len(log)*(1 - test_len)))

    df_train = log.iloc[:num_events]
    df_test = log.iloc[num_events:]

    # Incomplete final traces
    df_train = df_train.sort_values(by=['case_id', 'pos_trace'],
                                    ascending=True)
    inc_traces = pd.DataFrame(df_train.groupby('case_id')
                                .last()
                                .reset_index())
    inc_traces = inc_traces[inc_traces.pos_trace != inc_traces.trace_len]
    inc_traces = inc_traces['case_id'].to_list()

    # Drop incomplete traces
    df_test = df_test[~df_test.case_id.isin(inc_traces)]
    df_test = df_test.drop(columns=['trace_len', 'pos_trace'])

    df_train = df_train[~df_train.case_id.isin(inc_traces)]
    df_train = df_train.drop(columns=['trace_len', 'pos_trace'])
    key = 'end_timestamp' if one_timestamp else 'start_timestamp'
    df_test = (df_test
                .sort_values(key, ascending=True)
                .reset_index(drop=True).to_dict('records'))
    df_train = (df_train
                .sort_values(key, ascending=True)
                .reset_index(drop=True).to_dict('records'))
    df_test = pd.DataFrame(df_test)
    df_train = pd.DataFrame(df_train)
    return df_train, df_test

def split_data(PATH_LOG, column_names, PATH_LOG_test=None):
    file_name = os.path.splitext(os.path.basename(PATH_LOG))[0]
    file_extension = PATH_LOG.lower().split('.')[-1]
    if PATH_LOG_test == None:
        if file_extension == 'csv' or file_extension == 'gz':
            df = pd.read_csv(PATH_LOG)
        elif file_extension == 'xes':
            df = pm4py.read_xes(PATH_LOG)
        df = df.rename(columns=column_names)

        df_train_big, df_test = trainTestSplit(df=df, test_len=0.2)

    else:
        if file_extension == 'csv':
            df_train = pd.read_csv(PATH_LOG)
            df_test = pd.read_csv(PATH_LOG_test)
        elif file_extension == 'xes':
            df_train = pm4py.read_xes(PATH_LOG)
            df_test = pm4py.read_xes(PATH_LOG_test)
        df_train = df_train.rename(columns=column_names)
        df_test = df_test.rename(columns=column_names)

    # inform about number of cases
    print(f"The train log conisists of {len(set(df_train_big['case_id']))} cases")
    # print(f"The val log conisists of {len(set(df_val['case_id']))} cases")
    print(f"The test log conisists of {len(set(df_test['case_id']))} cases")

    data_dir = os.path.join(os.getcwd(), "input_data", file_name)
    if not os.path.exists(data_dir):
    # If it doesn't exist, create the directory
        os.makedirs(data_dir)
    train_path_big = os.path.join(data_dir,'train_log_large.csv')
    test_path = os.path.join(data_dir,'test_log.csv')
    df_train_big.to_csv(train_path_big, index=False)
    df_test.to_csv(test_path, index=False)

    number_test_cases = len(set(df_test['case_id']))

    return df_train_big, df_test, number_test_cases