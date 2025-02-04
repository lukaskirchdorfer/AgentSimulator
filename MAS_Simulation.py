import pandas as pd
import numpy as np
from scipy.stats import truncnorm
import scipy.stats as st
import math
from mesa import Agent, Model
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector
import random
import warnings
import sys, getopt, os
from datetime import datetime, timedelta
import pytz
import ast
import pm4py
import warnings
warnings.filterwarnings("ignore")

from source.train_test_split import split_data
from source.agent_types.discover_roles import discover_roles_and_calendars
from source.arrival_distribution import get_inter_arrival_times, get_best_fitting_distribution
from source.extraneous_delays.delay_discoverer import compute_complex_extraneous_activity_delays, compute_naive_extraneous_activity_delays
from source.extraneous_delays.event_log import EventLogIDs
from source.extraneous_delays.config import (
    Configuration as ExtraneousActivityDelaysConfiguration,
    TimerPlacement,
)
from source.agent_types.discover_resource_calendar import discover_calendar_per_agent
from source.petri_net_utils import discover_petri_net_inductive, write_petri_net, sample_next_activity, update_marking_after_transition, sample_next_role, get_stochastic_map, get_custom_stochastic_map
from source.agent_miner_code.generic_discovery import run_agent_miner
from source.interaction_probabilities import calculate_agent_handover_probabilities, calculate_agent_handover_probabilities_per_activity
from source.lstm import prepare_data_lstm, train_lstm_model, predict_next_step

STEPS_TAKEN = []

# helper functions
def preprocess(df):
    """
    Preprocess event log 
    """
    # transfrom case_id into int
    if df['case_id'].dtype == 'object':
        df['case_id'] = df['case_id'].str.extract('(\d+)').astype(int)

    # fill NaN values of resource column
    df['resource'].fillna('artificial', inplace=True)
    df['resource'] = df['resource'].astype(str)


    def rename_artificial(row):
        if row['resource'] == 'artificial':
            return row['resource'] + '_' + row['activity_name']
        else:
            return row['resource']

    # Apply the function to each row
    df['resource'] = df.apply(rename_artificial, axis=1)


    # name agents with plain integers 
    df['agent'] = pd.factorize(df['resource'])[0]

    # Create a mapping of integers to resource values
    integers = df['agent'].unique()
    resources = df['resource'].unique()
    integers_to_resources = dict(zip(integers, resources))
    # Create the agent_to_resource mapping dictionary by reversing the integers_to_resources dictionary
    AGENT_TO_RESOURCE = {k: v for k, v in integers_to_resources.items()}

    # insert a new row after every ending case
    df = insert_rows_before_case_change(df)

    return df, AGENT_TO_RESOURCE

def activities_with_zero_waiting_time(df, threshold=0.99):
    # Sort the DataFrame by start timestamp
    df_sorted = df.sort_values(by='start_timestamp')
    df_sorted = df_sorted.groupby('case_id')

    group_list = []

    for case_id, group in df_sorted:
        # Calculate waiting time for each group
        group['waiting_time'] = group['start_timestamp'] - group['end_timestamp'].shift(1)
        # Append the modified group to the list
        group_list.append(group)

    # Concatenate the list of groups back into a single DataFrame
    df_with_waiting_time = pd.concat(group_list)

    # replace NaT values (first activity per case) with 0
    df_with_waiting_time['waiting_time'] = df_with_waiting_time['waiting_time'].fillna(pd.Timedelta(seconds=0))
    
    # Filter rows where waiting time is 0
    zero_waiting_time_df = df_with_waiting_time[df_with_waiting_time['waiting_time'] <= pd.Timedelta(seconds=0)]
    
    # Group by activity name and count occurrences
    counts = zero_waiting_time_df.groupby('activity_name').size()
    
    # Get the total counts for each activity
    total_counts = df_with_waiting_time['activity_name'].value_counts()
    
    # Filter activities with count equal to total occurrences
    always_zero_waiting_time_activities = counts[counts >= total_counts[counts.index] * threshold]
    
    return always_zero_waiting_time_activities.index.tolist()

def sample_from_distribution(distribution):
    if distribution.type.value == "expon":
        scale = distribution.mean - distribution.min
        if scale < 0.0:
            print("Warning! Trying to generate EXPON sample with 'mean' < 'min', using 'mean' as scale value.")
            scale = distribution.mean
        sample = st.expon.rvs(loc=distribution.min, scale=scale, size=1)
    elif distribution.type.value == "gamma":
        # If the distribution corresponds to a 'gamma' with loc!=0, the estimation is done wrong
        # dunno how to take that into account
        sample = st.gamma.rvs(
            pow(distribution.mean, 2) / distribution.var,
            loc=0,
            scale=distribution.var / distribution.mean,
            size=1,
        )
    elif distribution.type.value == "norm":
        sample = st.norm.rvs(loc=distribution.mean, scale=distribution.std, size=1)
    elif distribution.type.value == "uniform":
        sample = st.uniform.rvs(loc=distribution.min, scale=distribution.max - distribution.min, size=1)
    elif distribution.type.value == "lognorm":
        # If the distribution corresponds to a 'lognorm' with loc!=0, the estimation is done wrong
        # dunno how to take that into account
        pow_mean = pow(distribution.mean, 2)
        phi = math.sqrt(distribution.var + pow_mean)
        mu = math.log(pow_mean / phi)
        sigma = math.sqrt(math.log(phi ** 2 / pow_mean))
        sample = st.lognorm.rvs(sigma, loc=0, scale=math.exp(mu), size=1)
    elif distribution.type.value == "fix":
        sample = [distribution.mean] * 1

    return sample[0]

def random_sample_timestamps_(date, min_time, max_time, x, arrival_distribution, first_day, start_timestamp):
    # Combine the provided date with the min and max times
    min_timestamp = pd.to_datetime(f'{date} {min_time}', utc=True)
    max_timestamp = pd.to_datetime(f'{date} {max_time}', utc=True)

    sampled_case_starting_times = []
    if first_day:
        time = start_timestamp
    else:
        time = min_timestamp
    sampled_case_starting_times.append(time)
    smaller_than_max_time = True
    while smaller_than_max_time:# and len(sampled_case_starting_times) < x:
        if arrival_distribution.type.value == "expon":
            scale = arrival_distribution.mean - arrival_distribution.min
            if scale < 0.0:
                print("Warning! Trying to generate EXPON sample with 'mean' < 'min', using 'mean' as scale value.")
                scale = arrival_distribution.mean
            sampled_times_between_cases = st.expon.rvs(loc=arrival_distribution.min, scale=scale, size=1)
        elif arrival_distribution.type.value == "gamma":
            sampled_times_between_cases = st.gamma.rvs(
                pow(arrival_distribution.mean, 2) / arrival_distribution.var,
                loc=0,
                scale=arrival_distribution.var / arrival_distribution.mean,
                size=1,
            )
        elif arrival_distribution.type.value == "norm":
            sampled_times_between_cases = st.norm.rvs(loc=arrival_distribution.mean, scale=arrival_distribution.std, size=1)
        elif arrival_distribution.type.value == "uniform":
            sampled_times_between_cases = st.uniform.rvs(loc=arrival_distribution.min, scale=arrival_distribution.max - arrival_distribution.min, size=1)
        elif arrival_distribution.type.value == "lognorm":
            pow_mean = pow(arrival_distribution.mean, 2)
            phi = math.sqrt(arrival_distribution.var + pow_mean)
            mu = math.log(pow_mean / phi)
            sigma = math.sqrt(math.log(phi ** 2 / pow_mean))
            sampled_times_between_cases = st.lognorm.rvs(sigma, loc=0, scale=math.exp(mu), size=1)
        elif arrival_distribution.type.value == "fix":
            sampled_times_between_cases = [arrival_distribution.mean] * 1
    
    
        for j in range(len(sampled_times_between_cases)):
            time = sampled_case_starting_times[-1] + pd.Timedelta(seconds=sampled_times_between_cases[j])
            # print(f"time: {time}")
            if time >= min_timestamp and time <= max_timestamp:
                sampled_case_starting_times.append(time)
            elif time > max_timestamp:
                smaller_than_max_time = False

    sampled_case_starting_times = sorted(sampled_case_starting_times)

    return sampled_case_starting_times

def increment_day_of_week(day_of_week):
    # Define a mapping of days of the week to their numerical representation
    days_mapping = {
        'MONDAY': 0, 'TUESDAY': 1, 'WEDNESDAY': 2,
        'THURSDAY': 3, 'FRIDAY': 4, 'SATURDAY': 5, 'SUNDAY': 6
    }

    # Get the numerical representation of the current day
    current_day_numeric = days_mapping[day_of_week]

    # Increment the day by one (circular increment, so Sunday wraps around to Monday)
    next_day_numeric = (current_day_numeric + 1) % 7

    # Get the corresponding day name for the incremented day
    next_day_of_week = [day for day, value in days_mapping.items() if value == next_day_numeric][0]

    return next_day_of_week

def get_min_max_time_per_day(case_start_timestamps):
    # get the min and max time of case arrivals per day of the week
    days_of_week = [day.strftime('%A').upper() for day in case_start_timestamps]
    days_of_week_set = set(days_of_week)
    min_max_time_per_day = {day: [pd.Timestamp('2023-01-01 23:59:59'),pd.Timestamp('2023-01-01 00:00:01')] for day in days_of_week_set}

    for i in range(len(case_start_timestamps)):
        for key, value in min_max_time_per_day.items():
            if days_of_week[i] == key:
                if case_start_timestamps[i].time() < value[0].time():
                    value[0] = case_start_timestamps[i]
                if case_start_timestamps[i].time() > value[1].time():
                    value[1] = case_start_timestamps[i]
    return min_max_time_per_day

def get_average_occurence_of_cases_per_day(case_start_timestamps):
    # get the mean and std of case arrivals per day of the week
    days_of_week = [day.strftime('%A').upper() for day in case_start_timestamps]
    days_of_week = set(days_of_week)
    count_occurrences_by_day = {day: [] for day in days_of_week}
    day = case_start_timestamps[0].strftime('%A').upper()
    counter = 0
    for time in case_start_timestamps:
        if day == time.strftime('%A').upper():
            counter += 1
        else:
            count_occurrences_by_day[day].append(counter)
            counter = 1
            day = time.strftime('%A').upper()

    average_occurrences_by_day = {day: () for day in days_of_week}
    for key, value in count_occurrences_by_day.items():
        average_occurrences_by_day[key] = (np.mean(value), np.std(value))

    return average_occurrences_by_day

def insert_rows_before_case_change(df):
    # Ensure the DataFrame is sorted by case_id and start_timestamp
    df.sort_values(by=['case_id', 'start_timestamp'], inplace=True)

    # Group by case_id and find the row with the latest start_timestamp in each group
    last_rows = df.groupby('case_id').tail(1)

    # Create a copy of the last row for each case and update the activity_name to 'zzz_end'
    end_rows = last_rows.copy()
    end_rows['activity_name'] = 'zzz_end'
    end_rows['start_timestamp'] = end_rows['end_timestamp']

    # Concatenate the original DataFrame with the new end_rows
    df_with_end_rows = pd.concat([df, end_rows], ignore_index=True)
    df_with_end_rows.sort_values(by=['case_id', 'start_timestamp'], inplace=True)
    df_with_end_rows.reset_index(drop=True, inplace=True)

    return df_with_end_rows

def compute_case_arrival_distribution(df):
    """
    computes mean and std for a normal distribution to sample case arrivals
    """
    df = df.sort_values(by='start_timestamp')
    case_id = 0
    # transform column to datetime
    case_start_times = []
    for i in range(len(df)):
        if df['case_id'][i] != case_id:
            case_start_times.append(df['start_timestamp'][i])
            case_id = df['case_id'][i]
    # compute time between cases in seconds
    time_between_cases = []
    for i in range(len(case_start_times)-1):
        time_between_cases.append((case_start_times[i+1] - case_start_times[i]).total_seconds())
    mean = np.mean(time_between_cases)
    std = np.std(time_between_cases)

    return mean, std

def compute_activity_duration_distribution(df):
    """
    computes mean and std for each activity duration in the log and returns this in form of a dict
    if no start times are available, they are set as the end time of the previous event
    start times of activities that start a case are estimated by taking their average duration when they do not start a case
    """
    activities = sorted(set(df['activity_name']))
    agents = sorted(set(df['agent']))
    # act_durations = {key: [] for key in activities}
    act_durations = {key: {k: [] for k in activities} for key in agents}
    df['end_timestamp']= pd.to_datetime(df['end_timestamp'], utc=True, format='mixed')
    if 'start_timestamp' in df.columns:
        df['start_timestamp']= pd.to_datetime(df['start_timestamp'], utc=True, format='mixed')
        for agent in agents:
            for activity in activities:
                for i in range(len(df)):
                    if df['agent'][i] == agent:
                        if df['activity_name'][i] == activity:
                            duration = (df['end_timestamp'][i] - df['start_timestamp'][i]).total_seconds()
                            act_durations[agent][activity].append(duration)
    else:
        case_id = 0
        df['start_timestamp'] = df['end_timestamp'] # init start timestamp column
        activities_without_start = []
        for i in range(len(df)):
            if df['case_id'][i] != case_id:
                df['start_timestamp'][i] = None
                case_id = df['case_id'][i]
                if df['activity_name'][i] not in activities_without_start:
                    activities_without_start.append(df['activity_name'][i])
                
            else:
                df['start_timestamp'][i] = df['end_timestamp'][i-1] # start time is end time of previous activity
        # check which activites have no start time
        available_durations = {key: [] for key in activities_without_start}
        # get activity durations for the activities that start a case based on their duration when they occur during a case
        for act in activities_without_start:
            for i in range(len(df)):
                if df['activity_name'][i] == act:
                    if pd.isnull(df['start_timestamp'][i]) == False:
                        available_durations[act].append((df['end_timestamp'][i] - df['start_timestamp'][i]).total_seconds())
        mean_available_durations = {key: np.mean(available_durations[key]) for key in available_durations}
        for act in activities_without_start:
            for i in range(len(df)):
                if df['activity_name'][i] == act:
                    if pd.isnull(df['start_timestamp'][i]) == True:
                        df['start_timestamp'][i] = df['end_timestamp'][i] - pd.Timedelta(seconds=mean_available_durations[act])

        for agent in agents:
            for activity in activities:
                for i in range(len(df)):
                    if df['agent'][i] == agent:
                        if df['activity_name'][i] == activity:
                            duration = (df['end_timestamp'][i] - df['start_timestamp'][i]).total_seconds()
                            act_durations[agent][activity].append(duration)

    return act_durations


# def compute_activity_duration_distribution_per_agent(activity_durations_dict):
#     """
#     Compute the best fitting distribution of activity durations per agent.

#     Args:
#         activity_durations_dict (dict): A dict storing lists of activity durations per agent from the training log.

#     Returns:
#         dict: A dict storing for each agent the distribution for each activity.
#     """
#     agents = activity_durations_dict.keys()
#     activities = []
#     for k,v in activity_durations_dict.items():
#         for kk, vv in v.items():
#             activities.append(kk)
#     activities = set(activities)

#     act_duration_distribution_per_agent = {agent: {act: [] for act in activities} for agent in agents}

#     for agent, val in activity_durations_dict.items():
#         for act, duration_list in val.items():
#             if len(duration_list) > 0:
#                 duration_distribution = get_best_fitting_distribution(
#                     data=duration_list,
#                     filter_outliers=True,
#                     outlier_threshold=20.0,
#                 )
#                 act_duration_distribution_per_agent[agent][act] = duration_distribution

#     return act_duration_distribution_per_agent


def _compute_activity_duration_distribution(df, res_calendars, roles):
    """
    computes mean and std for each activity duration in the log and returns this in form of a dict
    """
    # activities = sorted(set(df['activity_name']))
    # agents = sorted(set(df['agent']))
    # act_durations = {key: {k: [] for k in activities} for key in agents}
    # for agent in agents:
    #     for activity in activities:
    #         for i in range(len(df)):
    #             if df['agent'][i] == agent:
    #                 if df['activity_name'][i] == activity:
    #                     duration = (df['end_timestamp'][i] - df['start_timestamp'][i]).total_seconds()
    #                     act_durations[agent][activity].append(duration)

    # return act_durations

    activities = sorted(set(df['activity_name']))
    agents = sorted(set(df['agent']))
    act_durations = {key: {k: [] for k in activities} for key in agents}

    
    for agent in agents:
        if agent in res_calendars.keys():
            agent_calendar = res_calendars[agent].intervals_to_json()
        else:
            agent_calendar = next((ids['calendar'] for role, ids in roles.items() if agent in ids['agents']), None)
        # print(f"agent: {agent}")
        # print(f"agent_calendar: {agent_calendar}")
        # if agent_calendar is None:
        #     continue

        # Convert calendar to workday schedule
        work_schedule = {}
        for shift in agent_calendar:
            day = shift['from']  # e.g., 'MONDAY'
            start_time = pd.to_datetime(shift['beginTime']).time()  # e.g., '07:00:00'
            end_time = pd.to_datetime(shift['endTime']).time()  # e.g., '15:00:00'
            work_schedule[day] = (start_time, end_time)
        
        for activity in activities:
            # print(f"activity: {activity}")
            mask = (df['agent'] == agent) & (df['activity_name'] == activity)
            activity_events = df[mask]
            
            for _, event in activity_events.iterrows():
                start_time = event['start_timestamp']
                end_time = event['end_timestamp']
                # print(f"start_time: {start_time}, end_time: {end_time}")
                
                # Initialize counters
                total_duration = (end_time - start_time).total_seconds()
                off_time = 0
                current_time = start_time
                
                # Iterate through each day of the activity
                while current_time < end_time:
                    day_name = current_time.strftime('%A').upper()
                    # print(f"day_name: {day_name}")
                    day_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
                    day_end = day_start + pd.Timedelta(days=1)
                    
                    # Get work hours for this day
                    work_hours = work_schedule.get(day_name)
                    # print(f"work_hours: {work_hours}")
                    if work_hours:
                        work_start = day_start.replace(
                            hour=work_hours[0].hour,
                            minute=work_hours[0].minute,
                            second=work_hours[0].second
                        )
                        work_end = day_start.replace(
                            hour=work_hours[1].hour,
                            minute=work_hours[1].minute,
                            second=work_hours[1].second
                        )
                        
                        # Calculate off time for this day
                        day_activity_start = max(current_time, day_start)
                        day_activity_end = min(end_time, day_end)
                        
                        # Before work hours
                        if day_activity_start < work_start:
                            off_end = min(work_start, day_activity_end)
                            off_time += (off_end - day_activity_start).total_seconds()
                            # print(f"off time before work hours: {(off_end - day_activity_start).total_seconds()}")
                        
                        # After work hours
                        if day_activity_end > work_end:
                            off_start = max(work_end, day_activity_start)
                            off_time += (day_activity_end - off_start).total_seconds()
                            # print(f"off time after work hours: {(day_activity_end - off_start).total_seconds()}")
                    else:
                        # Full day off
                        day_activity_start = max(current_time, day_start)
                        day_activity_end = min(end_time, day_end)
                        off_time += (day_activity_end - day_activity_start).total_seconds()
                        # print(f"off time full day off: {(day_activity_end - day_activity_start).total_seconds()}")
                    # Move to next day
                    current_time = day_end

                # print(f"total_duration: {total_duration}, off_time: {off_time}")
                # Calculate actual working duration
                actual_duration = total_duration - off_time
                if actual_duration >= 0:  # Only add positive durations
                    act_durations[agent][activity].append(actual_duration)

    # # print the mean and std of the activity durations
    # for agent, activities in act_durations.items():
    #     for activity, durations in activities.items():
    #         if len(durations) > 0:
    #             print(f"agent: {agent}, activity: {activity}, mean: {np.mean(durations)}, std: {np.std(durations)}")
    
    # x=y

    return act_durations


def compute_activity_duration_distribution_per_agent(df_train, res_calendars, roles):
    """
    Compute the best fitting distribution of activity durations per agent.

    Args:
        df_train: Event log in pandas format

    Returns:
        dict: A dict storing for each agent the distribution for each activity.
    """
    activity_durations_dict = _compute_activity_duration_distribution(df_train, res_calendars, roles)

    agents = activity_durations_dict.keys()
    activities = []
    for k,v in activity_durations_dict.items():
        for kk, vv in v.items():
            activities.append(kk)
    activities = set(activities)

    act_duration_distribution_per_agent = {agent: {act: [] for act in activities} for agent in agents}

    for agent, val in activity_durations_dict.items():
        for act, duration_list in val.items():
            if len(duration_list) > 0:
                duration_distribution = get_best_fitting_distribution(
                    data=duration_list,
                    filter_outliers=True,
                    outlier_threshold=20.0,
                )
                act_duration_distribution_per_agent[agent][act] = duration_distribution

    return act_duration_distribution_per_agent


def compute_activity_duration_per_role(activity_durations_dict, roles):
    """
    compute activity duration mean and std for each role based on the mean durations per agent
    """
    roles_names = sorted(list(roles.keys()))
    activities = []
    for k,v in activity_durations_dict.items():
        for kk, vv in v.items():
            activities.append(kk)
    activities = set(activities)
    
    act_durations_role = {key: {k: [] for k in activities} for key in roles_names}
    for agent_id, values in activity_durations_dict.items():
        for activity, val in values.items():
            for role_name, vv in roles.items():
                if agent_id in vv['agents']:
                    if isinstance(val, list):
                        act_durations_role[role_name][activity].extend(val)


    act_duration_distribution_per_role = {key: {k: [] for k in activities} for key in roles_names}
    for role, acts in act_durations_role.items():
        for activity, duration_list in acts.items():
            if len(duration_list) > 0:
                duration_distribution = get_best_fitting_distribution(
                    data=duration_list,
                    filter_outliers=True,
                    outlier_threshold=20.0,
                )
                act_duration_distribution_per_role[role][activity] = duration_distribution

    # print(f"Dist per role: {act_duration_distribution_per_role}")
    for key, value in act_duration_distribution_per_role.items():
        for k, v in value.items():
            if not isinstance(v, list):
                pass
                # print(f"{key}: {k}, {v.type.value}, {v.mean}, {v.std}")
        
    return act_duration_distribution_per_role


# Function to create sequences of activities for each case
def create_sequences(df):
    df = df.sort_values(by=['case_id', 'start_timestamp'])
    sequences = []
    active_agents = []
    for case_id, group in df.groupby('case_id'):
        sequences.append(list(group['activity_name']))
        active_agents.append(list(group['agent']))
    return sequences, active_agents

# Function to create sequences of activities for each case
def create_sequences_global(df):
    df = df.sort_values(by=['case_id', 'start_timestamp'])
    sequences = []
    for case_id, group in df.groupby('case_id'):
        sequences.append(list(group['activity_name']))
    return sequences


def create_labeled_sequences(encoded_sequences, active_agents):
    max_window_size = max(len(seq) for seq in encoded_sequences)
    X = []  # Initialize an empty list to store input sequences
    y = []  # Initialize an empty list to store target sequences
    active_agent = []

    counter = 0
    # Iterate over each encoded sequence in the input list
    for seq in encoded_sequences:
        # Iterate over different window sizes
        for window_size in range(1, max_window_size + 1):
            # Iterate over each index up to len(seq) - window_size
            for i in range(len(seq) - window_size):
                # Append the window of size window_size to X
                X.append(tuple(seq[i:i + window_size]))
                # Append the element immediately following the window to y
                y.append(seq[i + window_size])
                active_agent.append(active_agents[counter][i + window_size-1])
        counter += 1
    return X, y, active_agent

def create_labeled_sequences_global(encoded_sequences):
    max_window_size = max(len(seq) for seq in encoded_sequences)
    X = []  # Initialize an empty list to store input sequences
    y = []  # Initialize an empty list to store target sequences

    # Iterate over each encoded sequence in the input list
    for seq in encoded_sequences:
        # Iterate over different window sizes
        for window_size in range(1, max_window_size + 1):
            # Iterate over each index up to len(seq) - window_size
            for i in range(len(seq) - window_size):
                # Append the window of size window_size to X
                X.append(tuple(seq[i:i + window_size]))
                # Append the element immediately following the window to y
                y.append(seq[i + window_size])
    return X, y

def compute_activity_transition_dict_global(business_process_data):
    sequences = create_sequences_global(business_process_data)

    sequence_parts, next_act = create_labeled_sequences_global(sequences)

    dict_ = {}
    for i in range(len(next_act)):
        if sequence_parts[i] not in dict_.keys():
            dict_[sequence_parts[i]] = {next_act[i]: 1}
        else:
            if next_act[i] not in dict_[sequence_parts[i]].keys():
                dict_[sequence_parts[i]][next_act[i]] = 1
            else:
                dict_[sequence_parts[i]][next_act[i]] += 1

    for k, v in dict_.items():
        sum_values = 0
        for key, value in v.items():
            sum_values += value
        for key, value in v.items():
            v[key] = value / sum_values

    return dict_


def compute_activity_transition_dict(business_process_data):
    sequences, active_agents = create_sequences(business_process_data)
    sequence_parts, next_act, active_agent = create_labeled_sequences(sequences, active_agents)

    # Initialize a nested dictionary to store transition probabilities per agent
    dict_ = {}

    for i in range(len(next_act)):
        # Extract the agent of the current sequence part
        agent = active_agent[i]

        # Check if the sequence part exists in the dictionary
        if sequence_parts[i] not in dict_.keys():
            dict_[sequence_parts[i]] = {agent: {next_act[i]: 1}}
        else:
            # Check if the agent exists for the sequence part
            if agent not in dict_[sequence_parts[i]].keys():
                dict_[sequence_parts[i]][agent] = {next_act[i]: 1}
            else:
                # Check if the next activity exists for the agent
                if next_act[i] not in dict_[sequence_parts[i]][agent].keys():
                    dict_[sequence_parts[i]][agent][next_act[i]] = 1
                else:
                    dict_[sequence_parts[i]][agent][next_act[i]] += 1

    # Normalize transition probabilities for each agent and sequence part
    for seq_part, agents in dict_.items():
        for agent, transitions in agents.items():
            sum_values = sum(transitions.values())
            for key, value in transitions.items():
                transitions[key] = value / sum_values

    return dict_


def get_prerequisites_per_activity(data, discover_parallel_work=True):
    # Check for parallel activities
    result = []
    # Group by case_id
    grouped = data.groupby('case_id')
    # Iterate over groups
    for case_id, group in grouped:
        # Call the function to check for parallel activities with a minimum of 2
        parallel_activities = check_parallel_activities(group, min_activities=2)
        
        # Extend the result list with the detected parallel activities
        result.extend([(case_id,) + tuple(activities) for activities in parallel_activities])

    parallel_activities = get_unique_parallel_activities(result=result)

    preceding_activities_dict = generate_preceding_activities_dict(data)
    # print(f"preceeding activties: {preceding_activities_dict}")
    # remove parallel activities as prerequisite for each other
    for i in range(len(parallel_activities)):
        for key, value in preceding_activities_dict.items():
            if key in parallel_activities[i]:
                par = parallel_activities[i]
                related_activities = [item for item in par if item != key]
                for j in related_activities:
                    if j in value: # TODO understand why we need to add this
                        value.remove(j)
            # join parallel activities in prerequisite of other activities to mark that they are both required
            parallels = parallel_activities[i]
            value_flattened = [item for sublist in value for item in sublist]

            if set(parallels).issubset(set(value_flattened)):
                # Check if 'parallels' is a subset of 'value_flattened'
                disjoint_part = list(set(parallels).symmetric_difference(set(value)))

                # Convert 'parallels' to a tuple before using it as a key
                parallels_tuple = tuple(parallels)

                # Use the tuple as a key in the dictionary
                preceding_activities_dict[key] = disjoint_part + [parallels_tuple]

    # remove the activity itself as prerequisite for itself
    new_dict = {key: [] for key, value in preceding_activities_dict.items()}
    for key, value in preceding_activities_dict.items():     
        for i in range(len(value)):
            if not isinstance(value[i], list):
                if not value[i] == key:
                    new_dict[key].append(value[i])

            # if value contains sublists
            else:
                if key in value[i]:
                    value[i].remove(key)
                    new_dict[key].append(value[i])
                else:
                    new_dict[key].append(value[i])

    preceding_activities_dict = new_dict

    if discover_parallel_work == False:
        parallel_activities = []

    return preceding_activities_dict, parallel_activities
    

def check_parallel_activities(group, min_activities=2):
    sorted_group = group.sort_values(by='start_timestamp')
    result = []

    # Iterate over the range of parallel activities (2 or more)
    for i in range(len(sorted_group) - min_activities + 1):
        current_end_time = sorted_group.iloc[i]['end_timestamp']
        parallel_activities = [sorted_group.iloc[j]['activity_name'] for j in range(i + 1, i + min_activities) if current_end_time > sorted_group.iloc[j]['start_timestamp']]

        if len(parallel_activities) == min_activities - 1:
            result.append(parallel_activities + [sorted_group.iloc[i]['activity_name']])

    return result

def get_unique_parallel_activities(result):
    activities_in_parallel = []
    if result:
        for item in result:
            activities = (item[1], item[2])
            activities_in_parallel.append((item[1], item[2]))
        activities_in_parallel = sorted(set(activities_in_parallel))
        activities_in_parallel = [list(set(item)) for item in activities_in_parallel]

        unique_tuples = set(tuple(sorted(inner_list)) for inner_list in activities_in_parallel)
        unique_list_of_lists = [list(unique_tuple) for unique_tuple in unique_tuples]
        unique_list_of_lists = [sublist for sublist in unique_list_of_lists if len(sublist) > 1]

        unique_list_of_lists = discover_connected_pairs(unique_list_of_lists)
    else:
        unique_list_of_lists = []

    return unique_list_of_lists

def discover_connected_pairs(activity_list):
    connections = []
    unconnected_pairs = []

    for pair in activity_list:
        connected_lists = []
        
        # Check if each entity in the current pair has other common pairs
        for entity in pair:
            entity_connections = []
            
            for other_pair in activity_list:
                if entity in other_pair and other_pair != pair:
                    entity_connections.extend(other_pair)

            connected_lists.append(entity_connections)

        # Check if both entities have a common pair
        if any(connected_lists):
            common_entities = set.intersection(*map(set, connected_lists))
            
            if common_entities:
                # Merge the pairs into a single list
                merged_pair = list(common_entities)
                merged_pair.extend(pair)
                connections.append(merged_pair)
        else:
            # If no connections, add the pair to unconnected pairs
            unconnected_pairs.append(pair)

        
    unique_set = set()

    for sublist in connections:
        tuple_sublist = tuple(sorted(sublist))
        unique_set.add(tuple_sublist)

    connections = [list(sublist) for sublist in unique_set]

    # get unconnected pairs
    long_list_connections = [item for sublist in connections for item in sublist]
    for pair in activity_list:
        for entity in pair:
            if entity in long_list_connections:
                pass
            else:
                unconnected_pairs.append(pair)

    unique_set = set()

    for sublist in unconnected_pairs:
        tuple_sublist = tuple(sorted(sublist))
        unique_set.add(tuple_sublist)

    unconnected_pairs = [list(sublist) for sublist in unique_set]

    connections.extend(unconnected_pairs)

    # add removed pairs again to connections
    for sublist_A in activity_list:
        if sublist_A not in connections:
            connections.append(sublist_A)
    # print(connections)

    return connections


def generate_preceding_activities_dict(data):
    preceding_activities_dict = {}

    # Group by case_id
    grouped = data.groupby('case_id')

    # Iterate over groups
    for case_id, group in grouped:
        sorted_group = group.sort_values(by='start_timestamp')

        # Iterate through the sorted activities
        for i in range(1, len(sorted_group)):
            current_activity = sorted_group.iloc[i]['activity_name']
            preceding_activity = sorted_group.iloc[i - 1]['activity_name']

            # Update the dictionary with the preceding activity
            if current_activity not in preceding_activities_dict:
                preceding_activities_dict[current_activity] = set()
            preceding_activities_dict[current_activity].add(preceding_activity)

    # Convert sets to lists
    preceding_activities_dict = {key: list(value) for key, value in preceding_activities_dict.items()}

    return preceding_activities_dict


def compute_concurrency_frequencies(df, parallel_activities_dict):
    df['start_timestamp'] = pd.to_datetime(df['start_timestamp'], format='mixed')
    df['end_timestamp'] = pd.to_datetime(df['end_timestamp'], format='mixed')
    # Group by case_id
    grouped = df.groupby('case_id')

    # define counters
    # parallels_counter_dict = {act: [0]*len(parallel_activities_dict[act]) for act in parallel_activities_dict.keys()}
    parallels_counter_dict = {act: {str(parallel_activities_dict[act][i]): 0 for i in range(len(parallel_activities_dict[act]))} for act in parallel_activities_dict.keys()}
    for key, value in parallels_counter_dict.items():
        parallels_counter_dict[key]['not parallel'] = 0

    # print(parallels_counter_dict)

    for case_id, group in grouped:
        sorted_group = group.sort_values(by='start_timestamp')
        sorted_group = sorted_group.reset_index(drop=True)
        # print(sorted_group)

        for key, value in parallel_activities_dict.items():
            # print(key)
            # print(value)
            # if the activity appears in the given case
            if key in sorted_group['activity_name'].values:
                # print("yes")
                for i in range(len(value)):
                    # if the parallel activity is a single one
                    if len(value[i]) == 1:
                        # print("len = 1")
                        # if the single parallel activity appears in the case as well
                        if value[i] in sorted_group['activity_name'].values:
                            # print("in activities")
                            # check if the activities happen in parallel
                            for k in range(len(sorted_group)):
                                # print(k)
                                # print(sorted_group['activity_name'][k])
                                # print(key)
                                # print(value[i][0])
                                # print(sorted_group['activity_name'])
                                if sorted_group['activity_name'][k] == key:
                                    # print("key times")
                                    start_A = sorted_group['start_timestamp'][k]
                                    end_A = sorted_group['end_timestamp'][k]
                                elif sorted_group['activity_name'][k] == value[i][0]:
                                    # print("value times")
                                    start_B = sorted_group['start_timestamp'][k]
                                    end_B = sorted_group['end_timestamp'][k]

                            if (start_A <= start_B and start_B < end_A) or (start_B <= start_A and start_A < end_B):
                                # print("Parallel")
                                # print(key)
                                # print(value[i])
                                # print(parallels_counter_dict)
                                parallels_counter_dict[key][str(value[i])] += 1
                            else:
                                # print("Not parallel")
                                parallels_counter_dict[key]['not parallel'] += 1
                        # parallel activity not in current case -> alone
                        else:
                            parallels_counter_dict[key]['not parallel'] += 1

                    # if there are multiple parallel activities that could be performed
                    else:
                        # print(f"len = {len(value[i])}")
                        # check if all these activities appear in the case as well
                        all_appear = True
                        for j in range(len(value[i])):
                            if value[i][j] not in sorted_group['activity_name'].values:
                                all_appear = False
                        # print(all_appear)
                        if all_appear:
                            max_start = None
                            min_end = None
                            # check if the maximum start time of the given activities is smaller than the minimum end time -> then they are all parallel
                            for k in range(len(sorted_group)):
                                # print(sorted_group['activity_name'][k])
                                # print(key)
                                # print(value[i])
                                if sorted_group['activity_name'][k] == key or sorted_group['activity_name'][k] in value[i]:
                                    # print("yes")
                                    if max_start == None:
                                        max_start = sorted_group['start_timestamp'][k]
                                    else:
                                        if sorted_group['start_timestamp'][k] > max_start:
                                            max_start = sorted_group['start_timestamp'][k]

                                    if min_end == None:
                                        min_end = sorted_group['end_timestamp'][k]
                                    else:
                                        if sorted_group['end_timestamp'][k] < min_end:
                                            min_end = sorted_group['end_timestamp'][k]

                            # print(max_start)
                            # print(min_end)
                            if max_start < min_end:
                                parallels_counter_dict[key][str(value[i])] += 1
                            else:
                                parallels_counter_dict[key]['not parallel'] += 1
                        else:
                            parallels_counter_dict[key]['not parallel'] += 1


    # print(parallels_counter_dict)
    # transform counts into probabilities
    parallels_probs_dict = parallels_counter_dict
    for key, value in parallels_counter_dict.items():
        total_frequencies = 0
        for k, v in value.items():
            total_frequencies += v
        for k, v in value.items():
            parallels_probs_dict[key][k] /= total_frequencies

    return parallels_probs_dict


def compute_cycle_time(df):
    df['start_timestamp'] = pd.to_datetime(df['start_timestamp'], format='mixed')
    df['end_timestamp'] = pd.to_datetime(df['end_timestamp'], format='mixed')

    # Calculate duration per case
    df['case_duration'] = df.groupby('case_id').apply(lambda x: (x['end_timestamp'].max() - x['start_timestamp'].min()).total_seconds() / 3600).reset_index(drop=True)

    # Compute average case duration
    average_duration = df['case_duration'].mean()

    return average_duration


def check_for_multitasking_per_resource(df, discover_multitask, resource_name='agent', start_timestamp='start_timestamp', end_timestamp='end_timestamp'):
    if discover_multitask:
        df = df.sort_values(by=[resource_name, start_timestamp])
        # Dictionary to store the count of multitasking and total tasks for each resource
        multitasking_counts = {}
        total_task_counts = df[resource_name].value_counts().to_dict()

        for i in range(len(df[end_timestamp]) - 1):
            cell_a = df.iloc[i][end_timestamp]
            cell_b_next = df.iloc[i+1][start_timestamp]

            if df.iloc[i][resource_name] == df.iloc[i+1][resource_name]:
                if cell_b_next < cell_a:
                    resource = df.iloc[i][resource_name]
                    # Increment multitasking count for the resource
                    multitasking_counts[resource] = multitasking_counts.get(resource, 0) + 1
                    # Increment total task count for the resource
                    # total_task_counts[resource] = total_task_counts.get(resource, 0) + 1

        # Calculate the ratio for each resource
        multitasking_ratio = {}
        for resource, total_task_count in total_task_counts.items():
            if resource in multitasking_counts.keys():
                multitasking_count = multitasking_counts[resource]
                ratio = multitasking_count / total_task_count if total_task_count > 0 else 0
            else:
                ratio = 0
            multitasking_ratio[resource] = ratio
    else:
        multitasking_ratio = {agent: 0.0 for agent in df[resource_name].unique()}

    return multitasking_ratio

def check_for_multitasking_number(df, resource_name='agent', start_timestamp='start_timestamp', end_timestamp='end_timestamp'):
    df = df.sort_values(by=[resource_name, start_timestamp])
    # Dictionary to store the maximum number of simultaneous activities for each resource
    max_simultaneous_activities = {}

    # Dictionary to store the count of activities being performed simultaneously for each resource
    simultaneous_counts = {}

    for i in range(len(df)):
        current_resource = df.iloc[i][resource_name]
        current_start = df.iloc[i][start_timestamp]
        current_end = df.iloc[i][end_timestamp]

        # Check if current resource is already being tracked for simultaneous activities
        if current_resource in simultaneous_counts:
            count = 0
            # Check if the current activity overlaps with any existing activities for the same resource
            for activity in simultaneous_counts[current_resource]:
                start = activity[0]
                end = activity[1]
                if not (current_end <= start or current_start >= end):
                    count += 1

            simultaneous_counts[current_resource].append((current_start, current_end))
            max_simultaneous_activities[current_resource] = max(max_simultaneous_activities.get(current_resource, 0), count + 1)

        else:
            simultaneous_counts[current_resource] = [(current_start, current_end)]
            max_simultaneous_activities[current_resource] = 1

    # print(simultaneous_counts)

    return max_simultaneous_activities


# end helper functions

class ResourceAgent(Agent):
    """
    One agent for each of the resources in the event log
    """
    def __init__(self, unique_id, model, resource, timer, contractor_agent=None):
        super().__init__(unique_id, model)
        self.resource = resource
        self.model = model
        self.is_busy = False
        self.is_busy_until = None
        self.contractor_agent = contractor_agent
        self.agent_type = next((role for role, ids in self.model.roles.items() if self.resource in ids['agents']), None)
        # print(f"agent {self.resource} is of type {self.agent_type}")
        # self.calendar = next((ids['calendar'] for role, ids in self.model.roles.items() if self.resource in ids['agents']), None)
        if self.resource in self.model.calendars.keys():
            self.calendar = self.model.calendars[self.resource].intervals_to_json()
        else:
            self.calendar = next((ids['calendar'] for role, ids in self.model.roles.items() if self.resource in ids['agents']), None)
            # print("role calendar")
            # print(self.calendar)
        self.timer = timer
        # print(f"agent {self.resource} has calendar {self.calendar}")
        self.occupied_times = []

    def step(self, last_possible_agent=False, parallel_activity=False, current_timestamp=None, perform_multitask=False):
        # if len(self.contractor_agent.case.additional_next_activities) < 1:
        if not parallel_activity:
            self.contractor_agent.current_activity_index = self.contractor_agent.new_activity_index
            activity = self.contractor_agent.activities[self.contractor_agent.current_activity_index]
            current_timestamp = current_timestamp

            activity_duration = self.contractor_agent.get_activity_duration(self.resource, activity)

            self.perform_task(current_timestamp, activity_duration, activity, last_possible_agent, perform_multitask=perform_multitask)

            self.contractor_agent.case.timestamp_before_and_gateway = current_timestamp
        else:
            additional_activity_index = self.model.additional_activity_index
            activity = self.contractor_agent.case.additional_next_activities[additional_activity_index]
            current_timestamp = current_timestamp
            activity_duration = self.contractor_agent.get_activity_duration(self.resource, activity)

            self.perform_task(current_timestamp, activity_duration, activity, last_possible_agent, additional_act=True, 
                              additional_agent_counter=additional_activity_index, perform_multitask=perform_multitask)

    def perform_task(self, current_timestamp, activity_duration, activity, last_possible_agent, additional_act=False, additional_agent_counter=0, perform_multitask=False):
        # print(f"current timestamp: {current_timestamp}")
        if activity in self.timer.keys():
            waiting_time_distribution = timers[activity]
            waiting_time = sample_from_distribution(distribution=waiting_time_distribution)
        else:
            waiting_time = 0
        current_timestamp += pd.Timedelta(seconds=waiting_time)
        # print(f"activity duration: {activity_duration}")


        # check if the activity can be performed in multi-tasking style
        if activity in self.model.activities_without_waiting_time:
            perform_multitask = True

        # check if agent is busy after updating availability status
        if self.is_busy_until != None:
            if self.is_busy_until <= current_timestamp:
                self.is_busy = False

        if self.is_occupied(current_timestamp, activity_duration) == False or perform_multitask == True:# or activity_duration == 0.0:
            # check if current timestamp lies within the availability of the agent
            if self.is_within_calendar(current_timestamp, activity_duration) or perform_multitask == True:
                # set as busy
                
                if activity_duration != 0.0:
                    self.occupied_times.append((current_timestamp, current_timestamp + pd.Timedelta(seconds=activity_duration)))

                    self.is_busy_until = current_timestamp + pd.Timedelta(seconds=activity_duration)
                    self.is_busy = True
                    self.model.agents_busy_until[self.resource] = self.is_busy_until
                else:
                    pass
                # advance current timestamp
                self.contractor_agent.case.current_timestamp = current_timestamp + pd.Timedelta(seconds=activity_duration)
                # add activity to case list to keep track of performed activities per case
                self.contractor_agent.case.add_activity_to_case(activity)
                self.contractor_agent.case.add_agent_to_case(self.resource)
                if self.model.lstm_model == None:
                    # advance marking in petri net
                    self.contractor_agent.case.current_role_net_marking, self.contractor_agent.case.currently_active_role_net = update_marking_after_transition(trans=self.contractor_agent.case.sampled_transition, 
                                                    net=self.contractor_agent.case.current_role_net, 
                                                    marking=self.contractor_agent.case.current_role_net_marking,
                                                    final_marking=self.contractor_agent.case.current_role_net_final_marking,)

                print(f"Activity performed: {activity} by {self.resource} in case {self.contractor_agent.case.case_id}")

                # set that activity is performed
                self.contractor_agent.activity_performed = True
                # set agent and role that performed activity
                self.contractor_agent.case.previous_agent = self.resource
                self.contractor_agent.case.previous_role = f"a{self.agent_type[-1]}"

    
                # remove activity from additional activities
                if additional_act == True:
                    index_to_delete = self.contractor_agent.case.additional_next_activities.index(activity)
                    self.contractor_agent.case.additional_next_activities.pop(index_to_delete)


                STEPS_TAKEN.append({'case_id': self.contractor_agent.case.case_id, 
                                    'agent': self.resource, 
                                    'activity_name': activity,
                                    'start_timestamp': current_timestamp,
                                    'end_timestamp': self.contractor_agent.case.current_timestamp,
                                    'TimeStep': self.model.schedule.steps,
                                    })
                
            elif self.start_time_in_calendar(current_timestamp):
                # print(f"simulate interruption")
                # end_time_without_interruptions = current_timestamp + pd.Timedelta(seconds=activity_duration)
                end_time = self.add_off_time_to_end_time(current_timestamp, activity_duration)
                self.occupied_times.append((current_timestamp, end_time))
                self.is_busy_until = end_time
                self.is_busy = True
                self.model.agents_busy_until[self.resource] = self.is_busy_until

                self.contractor_agent.case.current_timestamp = end_time
                self.contractor_agent.case.add_activity_to_case(activity)
                self.contractor_agent.activity_performed = True
                self.contractor_agent.case.previous_agent = self.resource

                STEPS_TAKEN.append({'case_id': self.contractor_agent.case.case_id, 
                                    'agent': self.resource, 
                                    'activity_name': activity,
                                    'start_timestamp': current_timestamp,
                                    'end_timestamp': self.contractor_agent.case.current_timestamp,
                                    'TimeStep': self.model.schedule.steps,
                                    })
                
            else:
                print(f"#######agent {self.resource} is free but time not within calendar")
                if last_possible_agent: # then increase timer by x seconds to try to get an available agent later
                    # move timestamp until agent is available again according to calendar
                    self.contractor_agent.case.current_timestamp = self.set_time_to_next_availability_when_not_in_calendar(current_timestamp, activity_duration)
                    # print(f"set timestamp until agent is available again: {self.contractor_agent.case.current_timestamp}")
                    if additional_act == True:
                        self.model.additional_activity_index += 1
                else:
                    pass # first try if one of the other possible agents is available
        else:
            print(f"agent {self.resource} is busy when trying to perform task {activity} until {self.is_busy_until}")
            if last_possible_agent: # then increase timer by x seconds to try to get an available agent later
                self.set_current_time_to_next_available_slot()
                if additional_act == True:
                    self.model.additional_activity_index += 1
            else:
                pass # first try if one of the other possible agents is available

    def is_occupied(self, new_start, activity_duration):
        new_end = new_start + pd.Timedelta(seconds=activity_duration)
        for start, end in self.occupied_times:
            if new_start < end and new_end > start:
                return True  # There is an overlap
        return False  # No overlap found
    
    def get_current_number_multitasking(self, new_start, activity_duration):
        new_end = new_start + pd.Timedelta(seconds=activity_duration)
        number_multitask = 1 # 1 because we have to add the current activity as well
        for start, end in self.occupied_times:
            if new_start < end and new_end > start:
                number_multitask += 1

        return number_multitask
    
    def set_current_time_to_next_available_slot(self,):
        new_time_set = False
        current_time = self.contractor_agent.case.current_timestamp
        self.occupied_times = sorted(self.occupied_times, key=lambda x: x[1])
        for start, end in self.occupied_times:
            if end > current_time:
                self.contractor_agent.case.current_timestamp = end
                new_time_set = True
                # print(f"moved time to: {end}")
                break
        if new_time_set == False:
            self.contractor_agent.case.current_timestamp += pd.Timedelta(seconds=60)
            # print(f"moved time by 60 seconds")

    def start_time_in_calendar(self, current_timestamp):
        day_of_week = current_timestamp.strftime('%A').upper()
        for entry in self.calendar:
            if entry['from'] == day_of_week:
                try:
                    begin_time = datetime.strptime(entry['beginTime'], '%H:%M:%S')
                except ValueError:
                    # If the first format fails, try the second format '%H:%M:%S.%f'
                    begin_time = datetime.strptime(entry['beginTime'], '%H:%M:%S.%f')

                begin_time_current_activity = datetime.combine(current_timestamp.date(), current_timestamp.time())
                begin_time_current_activity = begin_time_current_activity.time()

                if begin_time.time() <= begin_time_current_activity:
                    return True
        return False
        
    def is_within_calendar(self, current_timestamp, activity_duration):
        """
        check if the current timestamp + activtiy duration is within the availability calendar of the agent
        param current_timestamp: datetime object
        param activity_duration: duration of next activity in seconds

        return True or False
        """
        day_of_week = current_timestamp.strftime('%A').upper()
        end_time_of_activity = current_timestamp + pd.Timedelta(seconds=activity_duration)
        # print(f"expected end time of activity: {end_time_of_activity}")

        for entry in self.calendar:
            if entry['from'] == day_of_week:
                # Try parsing with the first format '%H:%M:%S'
                try:
                    begin_time = datetime.strptime(entry['beginTime'], '%H:%M:%S')
                except ValueError:
                    # If the first format fails, try the second format '%H:%M:%S.%f'
                    begin_time = datetime.strptime(entry['beginTime'], '%H:%M:%S.%f')
                try:
                    end_time = datetime.strptime(entry['endTime'], '%H:%M:%S')
                except ValueError:
                    # If the first format fails, try the second format '%H:%M:%S.%f'
                    end_time = datetime.strptime(entry['endTime'], '%H:%M:%S.%f')

                end_time_current_activity = datetime.combine(end_time_of_activity.date(), end_time_of_activity.time())
                end_time_current_activity = end_time_current_activity.time()

                begin_time_current_activity = datetime.combine(current_timestamp.date(), current_timestamp.time())
                begin_time_current_activity = begin_time_current_activity.time()

                # if begin_time.time() <= end_time_current_activity <= end_time.time():
                if begin_time.time() <= begin_time_current_activity:
                    if end_time_current_activity <= end_time.time():
                        return True

        return False
    
    def set_time_to_next_availability_when_not_in_calendar(self, current_timestamp, activity_duration):
        """
        Set current timestamp to the next availability according to resource calendar. 
        E.g., if current_timestamp=04:30, set it to 08:00
        """
        current_timestamp += pd.Timedelta(seconds=activity_duration)
        current_day = current_timestamp.strftime('%A').upper()  # Get the current day of the week
        # print(f"day of current timestamp: {current_day}")
        current_time = current_timestamp.time()

        # Find the working hours for the current day
        working_hours = None
        for day_schedule in self.calendar:
            if day_schedule['from'] == current_day and day_schedule['to'] == current_day:
                working_hours = (day_schedule['beginTime'], day_schedule['endTime'])
                break

        # If no working hours are defined for the current day, find the next working day
        if working_hours is None:
            next_working_day = current_timestamp
            next_working_hours = None

            while next_working_day.strftime('%A').upper() not in [day_schedule['from'] for day_schedule in self.calendar]:
                next_working_day += pd.Timedelta(days=1)

            for day_schedule in self.calendar:
                if day_schedule['from'] == next_working_day.strftime('%A').upper() and day_schedule['to'] == next_working_day.strftime('%A').upper():
                    next_working_hours = (day_schedule['beginTime'], day_schedule['endTime'])
                    break

            if next_working_hours is None:
                raise ValueError(f"No working hours defined for agent {self.resource} on any day.")

            # Set the timestamp to the beginning of the working hours on the next working day
            try:
                next_possible_timestamp = datetime.combine(next_working_day, datetime.strptime(next_working_hours[0], '%H:%M:%S').time())
            except ValueError:
                next_possible_timestamp = datetime.combine(next_working_day, datetime.strptime(next_working_hours[0], '%H:%M:%S%f').time())

            next_possible_timestamp = pd.Timestamp(next_possible_timestamp, tzinfo=pytz.UTC)
        else:
            # Parse the working hours to datetime objects
            try:
                begin_time = datetime.strptime(working_hours[0], '%H:%M:%S').time()
            except ValueError:
                begin_time = datetime.strptime(working_hours[0], '%H:%M:%S.%f').time()
            try:
                end_time = datetime.strptime(working_hours[1], '%H:%M:%S').time()
            except ValueError:
                end_time = datetime.strptime(working_hours[1], '%H:%M:%S.%f').time()

            # Check if the current timestamp is beyond the working hours of the current day
            if current_time > end_time:
                next_working_day = current_timestamp + pd.Timedelta(days=1)
            else:
                next_working_day = current_timestamp

            # Find the next available working day and working hours
            next_working_hours = None

            while next_working_day.strftime('%A').upper() not in [day_schedule['from'] for day_schedule in self.calendar]:
                next_working_day += pd.Timedelta(days=1)

            for day_schedule in self.calendar:
                if day_schedule['from'] == next_working_day.strftime('%A').upper() and day_schedule['to'] == next_working_day.strftime('%A').upper():
                    next_working_hours = (day_schedule['beginTime'], day_schedule['endTime'])
                    break

            if next_working_hours is None:
                raise ValueError(f"No working hours defined for agent {self.resource} on any day.")

            # Set the timestamp to the beginning of the working hours on the next working day
            try:
                next_possible_timestamp = datetime.combine(next_working_day, datetime.strptime(next_working_hours[0], '%H:%M:%S').time())
            except ValueError:
                next_possible_timestamp = datetime.combine(next_working_day, datetime.strptime(next_working_hours[0], '%H:%M:%S%f').time())
            next_possible_timestamp = pd.Timestamp(next_possible_timestamp, tzinfo=pytz.UTC) #tz='UTC')

        return next_possible_timestamp
    
    def add_off_time_to_end_time(self, start_time, activity_duration):
        """
        Calculates the actual end time by adding unavailable periods to the activity duration.
        
        Args:
            start_time (pd.Timestamp): The start time of the activity
            activity_duration (float): The duration of the activity in seconds
        
        Returns:
            pd.Timestamp: The end time including off-time periods
        """
        remaining_duration = activity_duration
        current_time = start_time
        
        while remaining_duration > 0:
            # Get the current day and time
            current_day = current_time.strftime('%A').upper()
            current_time_of_day = current_time.time()
            
            # Find working hours for the current day
            working_hours = None
            for entry in self.calendar:
                if entry['from'] == current_day:
                    try:
                        begin_time = datetime.strptime(entry['beginTime'], '%H:%M:%S').time()
                    except ValueError:
                        begin_time = datetime.strptime(entry['beginTime'], '%H:%M:%S.%f').time()
                    try:
                        end_time = datetime.strptime(entry['endTime'], '%H:%M:%S').time()
                    except ValueError:
                        end_time = datetime.strptime(entry['endTime'], '%H:%M:%S.%f').time()
                    working_hours = (begin_time, end_time)
                    break
            
            if working_hours is None:
                # No working hours today, move to next day at start time
                current_time = current_time + pd.Timedelta(days=1)
                current_time = current_time.replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                continue
            
            begin_time, end_time = working_hours
            
            if current_time_of_day < begin_time:
                # Before working hours, jump to beginning of working hours
                current_time = current_time.replace(
                    hour=begin_time.hour,
                    minute=begin_time.minute,
                    second=begin_time.second,
                    microsecond=begin_time.microsecond
                )
            elif current_time_of_day >= end_time:
                # After working hours, jump to next day
                current_time = current_time + pd.Timedelta(days=1)
                current_time = current_time.replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
            else:
                # During working hours, process duration until end of working hours
                time_until_end_of_day = (
                    datetime.combine(current_time.date(), end_time) - 
                    datetime.combine(current_time.date(), current_time_of_day)
                ).total_seconds()
                
                duration_to_process = min(remaining_duration, time_until_end_of_day)
                current_time += pd.Timedelta(seconds=duration_to_process)
                remaining_duration -= duration_to_process
                
                if remaining_duration > 0 and current_time.time() >= end_time:
                    # If we hit the end of working hours and still have duration left,
                    # move to the next day
                    current_time = current_time + pd.Timedelta(days=1)
                    current_time = current_time.replace(
                        hour=0, minute=0, second=0, microsecond=0
                    )
        
        return current_time

        

class ContractorAgent(Agent):
    """
    One contractor agent to assign tasks using the contraction net protocol
    """
    def __init__(self, unique_id, model, activities, transition_probabilities, agent_activity_mapping):
        super().__init__(unique_id, model)
        self.activities = activities
        self.transition_probabilities = transition_probabilities
        self.agent_activity_mapping = agent_activity_mapping
        self.model = model
        self.current_activity_index = None
        self.activity_performed = False
        self.next_agent_distribution = None

    def step(self, scheduler, agent_keys, cases):
        method = "step"
        agent_keys = agent_keys[1:] # exclude contractor agent here as we only want to perform resource agent steps
        # 1) sort by specialism
        # bring the agents in an order to first ask the most specialized agents to not waste agent capacity for future cases -> principle of specialization
        def get_key_length(key):
            return len(self.agent_activity_mapping[key])

        # Sort the keys using the custom key function
        if isinstance(agent_keys[0], list):
            sorted_agent_keys = []
            for agent_list in agent_keys:
                sorted_agent_keys.append(sorted(agent_list, key=get_key_length))
        else:
            sorted_agent_keys = sorted(agent_keys, key=get_key_length)
        # print(f"Agents sorted by specialism: {sorted_agent_keys}")
        
        # # 2) sort by next availability
        sorted_agent_keys = self.sort_agents_by_availability(sorted_agent_keys)
            
        # if self.model.central_orchestration == False:
        #         # 3) sort by transition probs
        #         current_agent = self.case.previous_agent
        #         if current_agent != -1:
        #             current_activity = self.case.activities_performed[-1]
        #             if current_agent in self.model.agent_transition_probabilities:
        #                 if current_activity in self.model.agent_transition_probabilities[current_agent]:
        #                     current_probabilities = self.model.agent_transition_probabilities[current_agent][current_activity]
        #                 else:
        #                     current_probabilities = self.model.agent_transition_probabilities[current_agent]
        #             sorted_agent_keys = sorted(sorted_agent_keys, key=lambda x: current_probabilities.get(x, 0), reverse=True)

        if self.model.central_orchestration == False:
            # 3) sort by transition probs
            current_agent = self.case.previous_agent
            if current_agent != -1:
                current_activity = self.case.activities_performed[-1]
                next_activity = self.activities[self.new_activity_index]  # Get the next activity
                
                # Navigate through the nested dictionary structure
                if current_agent in self.model.agent_transition_probabilities:
                    if current_activity in self.model.agent_transition_probabilities[current_agent]:
                        # Create a dictionary to store probabilities for each potential next agent
                        current_probabilities = {}
                        for agent in sorted_agent_keys:
                            # Sum up probabilities for the specific next activity across all agents
                            if agent in self.model.agent_transition_probabilities[current_agent][current_activity]:
                                prob = self.model.agent_transition_probabilities[current_agent][current_activity][agent].get(next_activity, 0)
                                current_probabilities[agent] = prob
                            else:
                                current_probabilities[agent] = 0
                        # print(f"current_activity: {current_activity}")
                        # print(f"current_agent: {current_agent}")
                        # print(f"current_probabilities: {self.model.agent_transition_probabilities[current_agent][current_activity]}")
                        # print(f"sorted_agent_keys before: {sorted_agent_keys}")
                        # Filter out agents with zero probability and sort remaining agents
                        sorted_agent_keys_ = [
                            agent for agent in sorted_agent_keys 
                            if current_probabilities.get(agent, 0) > 0
                        ]
                        if len(sorted_agent_keys_) > 0:
                            # sorted_agent_keys = sorted_agent_keys_
                            # sorted_agent_keys = sorted(sorted_agent_keys, 
                            #                      key=lambda x: current_probabilities.get(x, 0), 
                            #                      reverse=True)
                            probabilities = [current_probabilities[agent] for agent in sorted_agent_keys_]
                            sorted_agent_keys = random.choices(
                                sorted_agent_keys_,
                                weights=probabilities,
                                k=len(sorted_agent_keys_)
                            )
                        else:
                            sorted_agent_keys = sorted_agent_keys
                        # print(f"sorted_agent_keys after: {sorted_agent_keys}")                        
        # print(f"sorted_agent_keys: {sorted_agent_keys}")

        last_possible_agent = False

        if isinstance(sorted_agent_keys[0], list):
            for agent_key in sorted_agent_keys:
                for inner_key in agent_key:
                    if inner_key == agent_key[-1]:
                        last_possible_agent = True
                    if inner_key in scheduler._agents:
                        if self.activity_performed:
                            break
                        else:
                            current_timestamp = self.get_current_timestamp(inner_key, parallel_activity=True)
                            perform_multitask = False
                            getattr(scheduler._agents[inner_key], method)(last_possible_agent, 
                                                                          parallel_activity=True, current_timestamp=current_timestamp, perform_multitask=perform_multitask)
        else:
            for agent_key in sorted_agent_keys:
                if agent_key == sorted_agent_keys[-1]:
                    last_possible_agent = True
                if agent_key in scheduler._agents:
                    if self.activity_performed:
                        break
                    else:
                        current_timestamp = self.get_current_timestamp(agent_key)
                        perform_multitask = False
                        getattr(scheduler._agents[agent_key], method)(last_possible_agent, parallel_activity=False, current_timestamp=current_timestamp, perform_multitask=perform_multitask)
        self.activity_performed = False

    def sort_agents_by_availability(self, sorted_agent_keys):
        if isinstance(sorted_agent_keys[0], list):
            sorted_agent_keys_new = []
            for agent_list in sorted_agent_keys:
                sorted_agent_keys_new.append(sorted(agent_list, key=lambda x: self.model.agents_busy_until[x]))
        else:
            sorted_agent_keys_new = sorted(sorted_agent_keys, key=lambda x: self.model.agents_busy_until[x])

        return sorted_agent_keys_new
    
    
    def get_current_timestamp(self, agent_id, parallel_activity=False):
        if parallel_activity == False:
            current_timestamp = self.case.current_timestamp
        else:
            current_timestamp = self.case.timestamp_before_and_gateway

        return current_timestamp


    def get_activity_duration(self, agent, activity):
        activity_distribution = self.model.activity_durations_dict[agent][activity]
        if activity_distribution.type.value == "expon":
            scale = activity_distribution.mean - activity_distribution.min
            if scale < 0.0:
                print("Warning! Trying to generate EXPON sample with 'mean' < 'min', using 'mean' as scale value.")
                scale = activity_distribution.mean
            activity_duration = st.expon.rvs(loc=activity_distribution.min, scale=scale, size=1)[0]
        elif activity_distribution.type.value == "gamma":
            activity_duration = st.gamma.rvs(
                pow(activity_distribution.mean, 2) / activity_distribution.var,
                loc=0,
                scale=activity_distribution.var / activity_distribution.mean,
                size=1,
            )[0]
        elif activity_distribution.type.value == "norm":
            activity_duration = st.norm.rvs(loc=activity_distribution.mean, scale=activity_distribution.std, size=1)[0]
        elif activity_distribution.type.value == "uniform":
            activity_duration = st.uniform.rvs(loc=activity_distribution.min, scale=activity_distribution.max - activity_distribution.min, size=1)[0]
        elif activity_distribution.type.value == "lognorm":
            pow_mean = pow(activity_distribution.mean, 2)
            phi = math.sqrt(activity_distribution.var + pow_mean)
            mu = math.log(pow_mean / phi)
            sigma = math.sqrt(math.log(phi ** 2 / pow_mean))
            activity_duration = st.lognorm.rvs(sigma, loc=0, scale=math.exp(mu), size=1)[0]
        elif activity_distribution.type.value == "fix":
            activity_duration = activity_distribution.mean


        return activity_duration
    
    def sample_starting_activity(self):
        """
        Sample the activity that starts the case based on the frequency of starting activities in the train log
        """
        
        # Cache the start activities if not already cached
        if not hasattr(self, '_start_activities_dist'):
            # Get first activity for each case more efficiently
            df = self.model.data
            # Sort once and get first activity for each case
            first_activities = (df.sort_values(['case_id', 'start_timestamp', 'end_timestamp'])
                            .groupby('case_id')['activity_name']
                            .first())
            
            # Handle Start/start cases
            if "Start" in first_activities.values or "start" in first_activities.values:
                self._start_activities_dist = ("Start" if "Start" in first_activities.values else "start", None)
            else:
                # Calculate frequencies
                total_cases = len(df['case_id'].unique())
                start_count = first_activities.value_counts() / total_cases
                self._start_activities_dist = (list(start_count.index), list(start_count.values))

        # Use cached distribution
        activities, weights = self._start_activities_dist
        if isinstance(activities, str):  # Handle Start/start case
            sampled_activity = activities
        else:
            sampled_activity = random.choices(activities, weights=weights, k=1)[0]
        
        return sampled_activity
    
    def check_for_other_possible_next_activity(self, next_activity):
        possible_other_next_activities = []
        for key, value in self.model.prerequisites.items():
            for i in range(len(value)):
                # if values is a single list, then only ONE of the entries must have been performed already (XOR gateway)
                if not isinstance(value[i], list):
                    if value[i] == next_activity:
                        possible_other_next_activities.append(key)
                # if value contains sublists, then all of the values in the sublist must have been performed (AND gateway)
                else:
                    # if current next_activity contained in prerequisites
                    if any(next_activity in sublist for sublist in value[i]):
                        # if all prerequisites are fulfilled
                        if all(value_ in self.case.activities_performed for value_ in value[i]):
                            possible_other_next_activities.append(key)

        return possible_other_next_activities
    
    def check_for_concurrency(self, next_activity):
        possible_other_next_activities = []
        # print("Check for additional activities...")
        # determine possible activities and their respective probability to occur in parallel
        acts = []
        weights = []
        for key, value in self.model.parallels_probs_dict.items():
            if next_activity == key:
                for k, v in value.items():
                    # print(k)
                    if "[" in k:
                        activities = ast.literal_eval(k)
                    else:
                        activities = k
                    acts.append(activities)
                    weights.append(v)
        
        # sample the parallel activity/activities
        if next_activity in self.model.parallels_probs_dict.keys():
            possible_other_next_activities = random.choices(acts, weights, k=1)[0]
            for element in possible_other_next_activities:
                if any(element in sublist for sublist in self.case.activities_performed):
                    # print("At least one of the activities was already performed")
                    possible_other_next_activities = 'not parallel'
                    break
            else:
                pass

        return possible_other_next_activities
    
    def get_potential_agents(self, case):
        """
        check if there already happened activities in the current case
            if no: current activity is usual start activity
            if yes: current activity is the last activity of the current case
        """
        self.case = case
        case_ended = False

        current_timestamp = self.case.current_timestamp

        print(f"case id: {self.case.case_id}")

        if self.model.lstm_model != None:
            if case.get_last_activity() == None: # if first activity in case
                # sample starting activity
                sampled_start_act = self.sample_starting_activity()
                current_act = sampled_start_act
                self.new_activity_index = self.activities.index(sampled_start_act)
                next_activity = sampled_start_act
                # print(f"start activity: {next_activity}")
            else:
                next_activity, self.next_agent_distribution = predict_next_step(
                    model=self.model.lstm_model[0], 
                    activities=self.case.activities_performed, 
                    resources=self.case.agents_performed, 
                    activity_encoder=self.model.lstm_model[1], 
                    resource_encoder=self.model.lstm_model[2], 
                    )

            # check if next activity is zzz_end
            if next_activity == 'zzz_end':
                potential_agents = None
                case_ended = True
                return potential_agents, case_ended#, None, None
        else:
            # only sample new role if previous one is transitioned to final marking
            if self.case.currently_active_role_net == False:
                # step 1: sample next role from i-net and update net marking
                sampled_role = None
                not_end_of_i_net = True
                while sampled_role == None and not_end_of_i_net == True:
                    sampled_role = sample_next_role(net=self.case.i_net, marking=self.case.i_net_marking, probabilities=self.case.i_net_smap, current_role=self.case.previous_role)
                    # sampled_role = sample_next_role(net=self.case.i_net, marking=self.case.i_net_marking, smap=self.case.i_net_smap)
                    self.case.i_net_marking, not_end_of_i_net = update_marking_after_transition(trans=sampled_role, 
                                                    net=self.case.i_net, 
                                                    marking=self.case.i_net_marking,
                                                    final_marking=self.case.i_net_final_marking)
                    sampled_role = sampled_role.label
                    print(f"sampled role: {sampled_role}")

                if not_end_of_i_net == False:
                    print("case finished")
                    potential_agents = None
                    case_ended = True
                    return potential_agents, case_ended
                
                # set the current case's agent net accordingly
                self.case.current_role_net = self.case.role_nets[sampled_role]
                self.case.current_role_net_marking = self.case.role_nets_markings[sampled_role]
                self.case.current_role_net_final_marking = self.case.role_net_final_markings[sampled_role]
                if self.case.role_nets_smap != None:
                    self.case.current_role_net_smap = self.case.role_nets_smap[sampled_role]
                else:
                    self.case.current_role_net_smap = None


            # step 2: sample next activity from agent-net
            sampled_activity_label=None
            is_final_act = False
            while sampled_activity_label not in self.activities and is_final_act == False:
                print(f"is final activity: {is_final_act}")
                self.case.sampled_transition, is_final_act = sample_next_activity(net=self.case.current_role_net, 
                                        marking=self.case.current_role_net_marking, 
                                        final_marking=self.case.current_role_net_final_marking, 
                                        probabilities=self.case.current_role_net_smap
                                        )
                sampled_activity_label = self.case.sampled_transition.label
                print(f"sampled activity: {sampled_activity_label}")
                print(f"is final activity: {is_final_act}")
                # update marking if silent transition
                if sampled_activity_label == None and is_final_act == False:
                    self.case.current_role_net_marking, _ = update_marking_after_transition(trans=self.case.sampled_transition, 
                                                        net=self.case.current_role_net, 
                                                        marking=self.case.current_role_net_marking)
                elif sampled_activity_label == None and is_final_act == True:
                    self.case.currently_active_role_net = False
                    potential_agents = None
                    case_ended = False
                    return potential_agents, case_ended
                # elif sampled_activity_label in self.activities and is_final_act == True:
            next_activity = sampled_activity_label

        self.new_activity_index = self.activities.index(next_activity)
        
        # step 3: determine corresponding agents of that role being able to perform the activity
        # check which agents can potentially perform the next task
        potential_agents = [key for key, value in self.agent_activity_mapping.items() if any(next_activity == item for item in value)]

        if self.next_agent_distribution != None:
            # remove agents from next agent distribution that are not in potential agents
            potential_agents = [agent for agent in self.next_agent_distribution if agent in potential_agents]
        # also add contractor agent to list as he is always active
        potential_agents.insert(0, 9999)

        return potential_agents, case_ended
    

class Case:
    """
    represents a case, for example a patient in the medical surveillance process
    """
    def __init__(self, case_id, start_timestamp=None, agentminer_nets=None) -> None:
        self.case_id = case_id
        self.is_done = False
        self.activities_performed = []
        self.agents_performed = []
        self.case_start_timestamp = start_timestamp
        self.current_timestamp = start_timestamp
        self.additional_next_activities = []
        self.potential_additional_agents = []
        self.timestamp_before_and_gateway = start_timestamp
        self.previous_agent = -1
        self.previous_role = "-1"

        self.i_net = agentminer_nets[0][0]
        self.i_net_marking = agentminer_nets[0][1]
        self.i_net_final_marking = agentminer_nets[0][2]
        if agentminer_nets[0][3] == None:
            self.i_net_smap = None
        else:
            self.i_net_smap = agentminer_nets[0][3]
        self.role_nets = agentminer_nets[1][0]
        self.role_nets_markings = agentminer_nets[1][1]
        self.role_net_final_markings = agentminer_nets[1][2]
        if agentminer_nets[1][3] == None:
            self.role_nets_smap = None
        else:
            self.role_nets_smap = agentminer_nets[1][3]

        self.current_role_net = None
        self.current_role_net_marking = None
        self.current_role_net_final_marking = None

        self.currently_active_role_net = False
        self.reached_final_in_role_net = False
        

    def get_last_activity(self):
        """
        get last activity that happened in the current case
        """
        if len(self.activities_performed) == 0:
            return None
        else:
            return self.activities_performed[-1]
        
    def add_activity_to_case(self, activity):
        self.activities_performed.append(activity)

    def add_agent_to_case(self, agent):
        self.agents_performed.append(agent)
    
    def update_current_timestep(self, duration):
        self.current_timestamp += pd.Timedelta(seconds=duration)


class BusinessProcessModel(Model):
    def __init__(self, data, activity_durations_dict, sampled_case_starting_times, roles, calendars,
                 start_timestamp, agent_activity_mapping, transition_probabilities, prerequisites, 
                 parallel_activities, max_activity_count_per_case, parallels_probs_dict, timer, 
                 discover_parallel_work, multitasking_probs_per_resource, max_multitasking_activities, 
                 activities_without_waiting_time, agent_transition_probabilities, central_orchestration, 
                 agentminer_nets, lstm_model):
        self.data = data
        self.resources = sorted(set(self.data['agent']))
        activities = sorted(set(self.data['activity_name']))
        self.roles = roles
        self.agents_busy_until = {key: start_timestamp for key in self.resources}
        self.calendars = calendars

        self.agentminer_nets = agentminer_nets
        self.lstm_model = lstm_model

        self.activity_durations_dict = activity_durations_dict
        self.sampled_case_starting_times = sampled_case_starting_times
        self.past_cases = []
        self.maximum_case_id = 0

        self.prerequisites = prerequisites
        self.parallel_activities = parallel_activities
        self.max_activity_count_per_case = max_activity_count_per_case
        self.parallels_probs_dict = parallels_probs_dict

        self.timer = timer

        self.discover_parallel_work = discover_parallel_work
        self.multitasking_probs_per_resource = multitasking_probs_per_resource
        self.max_multitasking_activities = max_multitasking_activities
        self.activities_without_waiting_time = activities_without_waiting_time

        self.agent_transition_probabilities = agent_transition_probabilities

        self.central_orchestration = central_orchestration

        self.schedule = MyScheduler(self,)

        self.contractor_agent = ContractorAgent(unique_id=9999, model=self, activities=activities, transition_probabilities=transition_probabilities, agent_activity_mapping=agent_activity_mapping)
        self.schedule.add(self.contractor_agent)

        for agent_id in range(len(self.resources)):
            agent = ResourceAgent(agent_id, self, self.resources[agent_id], self.timer, self.contractor_agent)
            self.schedule.add(agent)

        # Data collector to track agent activities over time
        self.datacollector = DataCollector(agent_reporters={"Activity": "current_activity_index"})


    def step(self, cases):
        # check if there are still cases planned to arrive in the future
        if len(self.sampled_case_starting_times) > 1:
        # if there are still cases happening
            if cases:
                last_case = cases[-1]
                if last_case.current_timestamp >= self.sampled_case_starting_times[0]:
                    self.maximum_case_id += 1
                    new_case_id = self.maximum_case_id
                    new_case = Case(case_id=new_case_id, start_timestamp=self.sampled_case_starting_times[0], agentminer_nets=self.agentminer_nets)
                    cases.append(new_case)
                    # remove added case from sampled_case_starting_times list
                    self.sampled_case_starting_times = self.sampled_case_starting_times[1:]
            # if no cases are happening
            else:
                self.maximum_case_id += 1
                new_case_id = self.maximum_case_id
                new_case = Case(case_id=new_case_id, start_timestamp=self.sampled_case_starting_times[0], agentminer_nets=self.agentminer_nets)
                cases.append(new_case)
                # remove added case from sampled_case_starting_times list
                self.sampled_case_starting_times = self.sampled_case_starting_times[1:]
        # Sort cases by current timestamp
        cases.sort(key=lambda x: x.current_timestamp)
        # print(f"cases after sorting: {[case.current_timestamp for case in cases]}")
        # print("NEW SIMULATION STEP")
        for case in cases:
            current_active_agents, case_ended = self.contractor_agent.get_potential_agents(case=case)
            if case_ended:
                self.past_cases.append(case)
                cases.remove(case)
                if len(self.sampled_case_starting_times) == 1 and len(cases) == 0:
                    self.sampled_case_starting_times = self.sampled_case_starting_times[1:]
                continue
            if current_active_agents == None:
                continue # continue with next case
            else:
                current_active_agents_sampled = current_active_agents
                self.schedule.step(cases=cases, current_active_agents=current_active_agents_sampled)

            # print("##################")




class MyScheduler(BaseScheduler):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)

    def step(self, cases, current_active_agents=None):
        """
        Step through the agents, activating each agent in a dynamic subset.
        """
        # print(f"current active agents: {current_active_agents}")
        self.do_each(method="step", agent_keys=current_active_agents, cases=cases)
        self.steps += 1
        self.time += 1

    def get_agent_count(self):
        """
        Returns the current number of active agents in the model.
        """
        return len(self._agents)
    
    def do_each(self, method, cases, agent_keys=None, shuffle=False):
        # print(f"cases in do_each: {cases}")
        agent_keys_ = [agent_keys[0]] # only contractor agent
        if agent_keys_ is None:
            agent_keys_ = self.get_agent_keys()
        if shuffle:
            self.model.random.shuffle(agent_keys_)
        # print(f"agent keys: {agent_keys}")
        # print(f"agents: {self._agents}")
        for agent_key in agent_keys_:
            if agent_key in self._agents:
                getattr(self._agents[agent_key], method)(self, agent_keys, cases)


def catch_parameter(opt):
    """Change the captured parameters names"""
    switch = {'--log_path': 'log_path', '--train_path': 'train_path', '--test_path': 'test_path', '--case_id': 'case_id', '--activity_name': 'activity_name', 
              '--resource_name': 'resource_name', '--end_timestamp': 'end_timestamp', '--start_timestamp': 'start_timestamp', '--extr_delays': 'extr_delays', 
              '--parallel_work': 'parallel_work', '--multi_task': 'multi_task', '--central_orchestration': 'central_orchestration', '--i_net_transitions': 'i_net_transitions', 
              '--agent_net_transitions': 'agent_net_transitions', '--lstm': 'lstm'}
    return switch.get(opt) 

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    opts, args = getopt.getopt(sys.argv[1:], "", ["log_path=", "train_path=", "test_path=", "case_id=", "activity_name=", 
                                                  "resource_name=", "end_timestamp=", "start_timestamp=", "extr_delays=", 
                                                  "parallel_work=", "multi_task=", "central_orchestration=", "i_net_transitions=", 
                                                  "agent_net_transitions=", "lstm="])
    train_and_test = True
    column_names = {}
    discover_extr_delays = False
    discover_delays = False
    discover_parallel_work = False
    discover_multitask = False
    central_orchestration = False
    determine_automatically = True
    i_net_transitions = 'equal'
    agent_net_transitions = 'equal'
    use_lstm = False
    for opt, arg in opts:
        key = catch_parameter(opt)
        if key in ["log_path"]:
            PATH_LOG = arg
            train_and_test = False
        else:
            if key in ["train_path"]:
                PATH_LOG = arg
            elif key in ["test_path"]:
                PATH_LOG_test = arg
        if key in ["case_id"]:
            column_names[arg] = 'case_id'
        elif key in ["activity_name"]:
            column_names[arg] = "activity_name"
        elif key in ["resource_name"]:
            column_names[arg] = 'resource'
        elif key in ["end_timestamp"]:
            column_names[arg] = 'end_timestamp'
        elif key in ["start_timestamp"]:
            column_names[arg] = 'start_timestamp'

        if key in ["extr_delays"]:
            if arg == "True":
                discover_extr_delays = True
                discover_delays = True
        if key in ["central_orchestration"]:
            if arg == "True":
                central_orchestration = True
            determine_automatically = False
        if key in ["i_net_transitions"]:
            i_net_transitions = arg
        if key in ["agent_net_transitions"]:
            agent_net_transitions = arg
        if key in ["lstm"]:
            if arg == "global":
                use_lstm = 'global'
            elif arg == "local":
                use_lstm = 'local'
            

    file_name = os.path.splitext(os.path.basename(PATH_LOG))[0]
    if determine_automatically:
        print("Choice for architecture and extraneous delays will be determined automatically")
        file_name_extension = 'main_results'
    else:
        if central_orchestration:
            if use_lstm != False:
                if use_lstm == 'global':
                    file_name_extension = f"orchestrated_lstm"
                elif use_lstm == 'local':
                    file_name_extension = f"orchestrated_lstm_local2"
            else:
                file_name_extension = f"orchestrated_{i_net_transitions}_{agent_net_transitions}"
        else:
            if use_lstm != False:
                if use_lstm == 'global':
                    file_name_extension = f"autonomous_lstm"
                elif use_lstm == 'local':
                    file_name_extension = f"autonomous_lstm_local2"
            else:
                file_name_extension = f"autonomous_{i_net_transitions}_{agent_net_transitions}"
    if train_and_test:
        business_process_data, df_test, num_cases_to_simulate = split_data(PATH_LOG, column_names, PATH_LOG_test)
    else:
        business_process_data, df_test, num_cases_to_simulate = split_data(PATH_LOG, column_names)

    def extract_last_20_percent(df):
        df_sorted = df.sort_values(by=['case_id', 'start_timestamp'])
        total_cases = df_sorted['case_id'].nunique()
        twenty_percent = int(total_cases * 0.2)
        last_20_percent_case_ids = df_sorted['case_id'].unique()[-twenty_percent:]
        df_val = df_sorted[df_sorted['case_id'].isin(last_20_percent_case_ids)]
        
        return df_val
    
    df_val = extract_last_20_percent(business_process_data)

    num_cases_to_simulate_val = len(set(df_val['case_id']))

    # preprocess data and get roles
    business_process_data, AGENT_TO_RESOURCE = preprocess(business_process_data)
    df_test, _ = preprocess(df_test)
    df_val, _ = preprocess(df_val)

    # compute mean and std of activity durations
    # activity_durations_dict = compute_activity_duration_distribution(business_process_data)
    # _ = compute_activity_duration_distribution(df_test)
    # _ = compute_activity_duration_distribution(df_val)
    business_process_data['end_timestamp']= pd.to_datetime(business_process_data['end_timestamp'], utc=True, format='mixed')
    business_process_data['start_timestamp']= pd.to_datetime(business_process_data['start_timestamp'], utc=True, format='mixed')
    df_test['end_timestamp']= pd.to_datetime(df_test['end_timestamp'], utc=True, format='mixed')
    df_test['start_timestamp']= pd.to_datetime(df_test['start_timestamp'], utc=True, format='mixed')
    df_val['end_timestamp']= pd.to_datetime(df_val['end_timestamp'], utc=True, format='mixed')
    df_val['start_timestamp']= pd.to_datetime(df_val['start_timestamp'], utc=True, format='mixed')

    data_dir = os.path.join(os.getcwd(), "simulated_data", file_name, file_name_extension)
    print(data_dir)
    os.system(f"mkdir {data_dir}")
    if not os.path.exists(data_dir):
    # If it doesn't exist, create the directory
        os.makedirs(data_dir)
    # save train data
    path_to_train_file = os.path.join(data_dir,"train_preprocessed.csv")
    path_to_train_file_end = os.path.join(data_dir,"train_preprocessed_end.csv")
    business_process_data_without_end_activity = business_process_data.copy()
    business_process_data.to_csv(path_to_train_file_end, index=False)
    business_process_data_without_end_activity = business_process_data_without_end_activity[business_process_data_without_end_activity['activity_name'] != 'zzz_end']
    business_process_data_without_end_activity.to_csv(path_to_train_file, index=False)
    print(f"train data saved to {path_to_train_file}")

    # save test data
    path_to_test_file = os.path.join(data_dir,"test_preprocessed.csv")
    df_test_without_end_activity = df_test.copy()
    df_test_without_end_activity = df_test_without_end_activity[df_test_without_end_activity['activity_name'] != 'zzz_end']
    df_test_without_end_activity.to_csv(path_to_test_file, index=False)


    # discover petri net
    # net, initial_marking, final_marking, smap = discover_petri_net_inductive(log_df=business_process_data_without_end_activity)
    # pn_path = os.path.join(data_dir,"petri_net.pnml")
    # write_petri_net(net, initial_marking, final_marking, pn_path)

    # get activities with 0 waiting time
    activities_without_waiting_time = activities_with_zero_waiting_time(business_process_data)

    # each simulated log starts at the time when the last case of the validation log starts
    # start_timestamp = max(business_process_data.groupby('case_id')['start_timestamp'].min().to_list())
    start_timestamp = min(df_test.groupby('case_id')['start_timestamp'].min().to_list())
    start_timestamp_val = min(df_val.groupby('case_id')['start_timestamp'].min().to_list())
    # print(f"######## start timestamp for simulation: {start_timestamp}")

    # extract roles and calendars  
    roles = discover_roles_and_calendars(business_process_data_without_end_activity)
    print(f"roles: {roles}")

    res_calendars, task_resources, joint_resource_events, pools_json, coverage_map = discover_calendar_per_agent(business_process_data_without_end_activity)

    # compute activity durations per agent
    # activity_durations_dict = compute_activity_duration_per_role(activity_durations_dict, roles)
    # activity_durations_dict = compute_activity_duration_distribution_per_agent(activity_durations_dict)
    activity_durations_dict = compute_activity_duration_distribution_per_agent(business_process_data_without_end_activity, res_calendars, roles)


    if use_lstm != False:
        X_activities, X_resources, y_activity, y_resource, activity_encoder, resource_encoder = prepare_data_lstm(business_process_data)
        activity_vocab_size = len(activity_encoder.classes_)  # Number of unique activities
        resource_vocab_size = len(resource_encoder.classes_)   # Number of unique resources
        # Train the LSTM model
        if use_lstm == 'global':
            agent_centric = False
            resource_encoder = None
        else:
            agent_centric = True
        model = train_lstm_model(X_activities, X_resources, y_activity, y_resource, activity_vocab_size, resource_vocab_size, agent_centric)

        lstm_model = (model, activity_encoder, resource_encoder)

        agentminer_nets = ((None, None, None, None), (None, None, None, None))
    else:
        lstm_model = None

        # discover agentminer nets
        path_to_agentminer_nets = run_agent_miner(data_dir=data_dir,
                        file_name='train_preprocessed.csv',
                        roles=roles)
        pn_path = path_to_agentminer_nets
        # read interaction net
        i_net, i_net_marking, i_net_final_marking = pm4py.read_pnml(os.path.join(pn_path, 'i-net-aol.pnml'))
        # read agent nets
        # parent_parent_dir = os.path.basename(os.path.dirname(os.path.dirname(pn_path)))
        # Extract and return the last character of the parent-parent directory
        # number_roles = int(parent_parent_dir[-1])
        number_roles = len(roles.keys())
        role_nets = {}
        role_nets_markings = {}
        role_net_final_markings = {}
        for i in range(1, number_roles+1):
            role = 'a' + str(i)
            role_nets[role], role_nets_markings[role], role_net_final_markings[role] = pm4py.read_pnml(os.path.join(pn_path, 'agents', "('a"+str(i)+"',)_agent-net-aol.pnml"))
        

        if i_net_transitions == 'smap':
            interaction_log = pd.read_csv(os.path.join(pn_path, '_log_in.csv'))
            i_net_smap = get_stochastic_map(
                log=interaction_log, 
                net=i_net, 
                initial_marking=i_net_marking, 
                final_marking=i_net_final_marking)
        elif i_net_transitions == 'custom':
            interaction_log = pd.read_csv(os.path.join(pn_path, '_log_in.csv'))
            # i_net_smap = get_interaction_net_probabilities(interaction_log)
            i_net_path = os.path.join(pn_path, 'i-net-aol.pnml')
            i_net_smap = get_custom_stochastic_map(interaction_log, i_net, i_net_marking, i_net_final_marking, i_net_path)
        else:
            i_net_smap = None

        if agent_net_transitions == 'smap':
            role_nets_smap = {}
            for i in range(1, number_roles+1):
                role = 'a' + str(i)
                agent_log = pd.read_csv(os.path.join(pn_path, 'agents', "('a"+str(i)+"',)_log.csv"))
                role_nets_smap[role] = get_stochastic_map(
                    log=agent_log, 
                    net=role_nets[role], 
                    initial_marking=role_nets_markings[role], 
                    final_marking=role_net_final_markings[role])
        elif agent_net_transitions == 'custom':
            role_nets_smap = {}
            for i in range(1, number_roles+1):
                role = 'a' + str(i)
                agent_log = pd.read_csv(os.path.join(pn_path, 'agents', "('a"+str(i)+"',)_log.csv"))
                role_nets_smap[role] = get_custom_stochastic_map(agent_log, role_nets[role], role_nets_markings[role], role_net_final_markings[role], os.path.join(pn_path, 'agents', "('a"+str(i)+"',)_agent-net-aol.pnml"))
        else:
            role_nets_smap = None

        agentminer_nets = ((i_net, i_net_marking, i_net_final_marking, i_net_smap), (role_nets, role_nets_markings, role_net_final_markings, role_nets_smap))

    # Case Arrival Distribution
    inter_arrival_durations = get_inter_arrival_times(business_process_data)
    # Get the best distribution fitting the inter-arrival durations
    arrival_distribution = get_best_fitting_distribution(
        data=inter_arrival_durations,
        filter_outliers=False,
        outlier_threshold=20.0,
    )

    case_start_timestamps = business_process_data.groupby('case_id')['start_timestamp'].min().tolist()
    min_max_time_per_day = get_min_max_time_per_day(case_start_timestamps)
    average_occurrences_by_day = get_average_occurence_of_cases_per_day(case_start_timestamps)

    date_of_current_timestamp = start_timestamp
    day_of_current_timestamp = date_of_current_timestamp.strftime('%A').upper()
    sampled_cases = []

    first_day = True

    while len(sampled_cases) <= num_cases_to_simulate:
        date_string = date_of_current_timestamp.strftime('%Y-%m-%d')
        if day_of_current_timestamp in min_max_time_per_day.keys():
            # date_string = start_timestamp.strftime('%Y-%m-%d')
            min_timestamp = min_max_time_per_day[day_of_current_timestamp][0].time().strftime('%H:%M:%S')
            max_timestamp = min_max_time_per_day[day_of_current_timestamp][1].time().strftime('%H:%M:%S')
            x = round(average_occurrences_by_day[day_of_current_timestamp][0])

            times = random_sample_timestamps_(date_string, min_timestamp, max_timestamp, x, 
                                              arrival_distribution, first_day, start_timestamp)
            first_day = False

            # sampled_cases.append(times)
            sampled_cases.extend(times)
            # sampled_cases = [item for sublist in sampled_cases for item in sublist]

            day_of_current_timestamp = increment_day_of_week(day_of_current_timestamp)
            date_of_current_timestamp += pd.Timedelta(days=1)
        
        else:
            day_of_current_timestamp = increment_day_of_week(day_of_current_timestamp)
            date_of_current_timestamp += pd.Timedelta(days=1)

    sampled_cases = sampled_cases[:num_cases_to_simulate + 1]

    ### do the same for validation
    date_of_current_timestamp_val = start_timestamp_val
    day_of_current_timestamp_val = date_of_current_timestamp_val.strftime('%A').upper()
    sampled_cases_val = []

    first_day = True

    while len(sampled_cases_val) <= num_cases_to_simulate_val:
        date_string = date_of_current_timestamp_val.strftime('%Y-%m-%d')
        if day_of_current_timestamp_val in min_max_time_per_day.keys():
            # date_string = start_timestamp.strftime('%Y-%m-%d')
            min_timestamp = min_max_time_per_day[day_of_current_timestamp_val][0].time().strftime('%H:%M:%S')
            max_timestamp = min_max_time_per_day[day_of_current_timestamp_val][1].time().strftime('%H:%M:%S')
            x = round(average_occurrences_by_day[day_of_current_timestamp_val][0])

            times = random_sample_timestamps_(date_string, min_timestamp, max_timestamp, x, 
                                              arrival_distribution, first_day, start_timestamp_val)
            first_day = False

            # sampled_cases.append(times)
            sampled_cases_val.extend(times)
            # sampled_cases = [item for sublist in sampled_cases for item in sublist]

            day_of_current_timestamp_val = increment_day_of_week(day_of_current_timestamp_val)
            date_of_current_timestamp_val += pd.Timedelta(days=1)
        
        else:
            day_of_current_timestamp_val = increment_day_of_week(day_of_current_timestamp_val)
            date_of_current_timestamp_val += pd.Timedelta(days=1)

    sampled_cases_val = sampled_cases_val[:num_cases_to_simulate_val + 1]

    #####

    # Some further mining steps
    # define mapping of agents to activities based on event log
    agent_activity_mapping = business_process_data.groupby('agent')['activity_name'].unique().apply(list).to_dict()

    transition_probabilities_autonomous = compute_activity_transition_dict(business_process_data)
    agent_transition_probabilities_autonomous = calculate_agent_handover_probabilities_per_activity(business_process_data)
    agent_transition_probabilities = None
    transition_probabilities = compute_activity_transition_dict_global(business_process_data)


    # get prerequisites for each activity
    prerequisites, parallel_activities = get_prerequisites_per_activity(business_process_data, discover_parallel_work)
    parallel_activities_list = [element for sublist in parallel_activities for element in sublist]
    parallel_activities_dict = {act: [] for act in parallel_activities_list}
    for pair in parallel_activities:
        for act in pair:
            parallels = [element for element in pair if element != act]
            parallel_activities_dict[act].append(parallels)

    parallels_probs_dict = compute_concurrency_frequencies(business_process_data, parallel_activities_dict)

    # get maximum activity frequency per case
    # Group by case_id and activity_name, then count occurrences
    activity_counts = business_process_data.groupby(['case_id', 'activity_name']).size().reset_index(name='count')
    # Find the maximum count for each activity across all cases
    max_activity_count_per_case = activity_counts.groupby('activity_name')['count'].max().to_dict()

    multitasking_probs_per_resource = check_for_multitasking_per_resource(business_process_data, discover_multitask)
    max_multitasking_activities = check_for_multitasking_number(business_process_data)

    def get_times_for_extrt_delays(discover_extr_delays=True):
        if discover_extr_delays == True:
            discovery_method = "complex"
            # Set-up configuration for extraneous delay discovery
            configuration = ExtraneousActivityDelaysConfiguration(
                log_ids=EventLogIDs(),
                # process_name=self.event_log.process_name,
                num_iterations=1,
                num_evaluation_simulations=3,
                training_partition_ratio=0.5,
                optimization_metric="relative_emd",
                discovery_method="eclipse-aware",
                timer_placement=TimerPlacement.BEFORE,
            )

            # print("defined configuration")

            if discovery_method == "naive":
                timers = compute_naive_extraneous_activity_delays(
                    business_process_data,
                    configuration,
                    configuration.should_consider_timer,
                    )
            elif discovery_method == "complex":
                timers = compute_complex_extraneous_activity_delays(
                    business_process_data,
                    configuration,
                    configuration.should_consider_timer,
                )
        else:
            timers = {}

        return timers
    
    timers_extr = get_times_for_extrt_delays(discover_extr_delays=True)
    timers = get_times_for_extrt_delays(discover_extr_delays=False)

    if determine_automatically:   
        # 1) simulate val log with extr delays and central orchestration
        discover_extr_delays = True
        central_orchestration = True
        # timers_extr = get_times_for_extrt_delays(discover_extr_delays)
        timers = timers_extr
        sampled_case_starting_times = sampled_cases_val
        start_timestamp = sampled_case_starting_times[0]
        sampled_case_starting_times = sampled_case_starting_times[1:]

        # Create the model using the loaded data
        business_process_model = BusinessProcessModel(business_process_data, activity_durations_dict, 
                                                        sampled_case_starting_times, roles, res_calendars, start_timestamp,
                                                        agent_activity_mapping, transition_probabilities, prerequisites, parallel_activities,
                                                        max_activity_count_per_case, parallels_probs_dict, timers, discover_parallel_work, 
                                                        multitasking_probs_per_resource, max_multitasking_activities, activities_without_waiting_time,
                                                        agent_transition_probabilities, central_orchestration, agentminer_nets, lstm_model)

        # define list of cases
        case_id = 0
        case_ = Case(case_id=case_id, start_timestamp=start_timestamp, agentminer_nets=agentminer_nets) # first case
        cases = [case_]

        # Run the model for a specified number of steps
        while business_process_model.sampled_case_starting_times: # while cases list is not empty
            business_process_model.step(cases)
        simulated_log_val_extr = pd.DataFrame(STEPS_TAKEN)
        STEPS_TAKEN = [] # reset

        # 2) simulate val log without extr delays and central orchestration
        discover_extr_delays = False
        central_orchestration = True
        timers = get_times_for_extrt_delays(discover_extr_delays)
        sampled_case_starting_times = sampled_cases_val
        start_timestamp = sampled_case_starting_times[0]
        sampled_case_starting_times = sampled_case_starting_times[1:]

        # Create the model using the loaded data
        business_process_model = BusinessProcessModel(business_process_data, activity_durations_dict, 
                                                        sampled_case_starting_times, roles, res_calendars, start_timestamp,
                                                        agent_activity_mapping, transition_probabilities, prerequisites, parallel_activities,
                                                        max_activity_count_per_case, parallels_probs_dict, timers, discover_parallel_work, 
                                                        multitasking_probs_per_resource, max_multitasking_activities, activities_without_waiting_time,
                                                        agent_transition_probabilities, central_orchestration, agentminer_nets, lstm_model)

        # define list of cases
        case_id = 0
        case_ = Case(case_id=case_id, start_timestamp=start_timestamp, agentminer_nets=agentminer_nets) # first case
        cases = [case_]

        # Run the model for a specified number of steps
        while business_process_model.sampled_case_starting_times: # while cases list is not empty
            business_process_model.step(cases)
        simulated_log_val_ = pd.DataFrame(STEPS_TAKEN)
        print(f"number of simulated cases: {len(business_process_model.past_cases)}")
        STEPS_TAKEN = [] # reset

        # 3) simulate val log with extr delays and autonomous handover
        discover_extr_delays = True
        central_orchestration = False
        # timers_extr = get_times_for_extrt_delays(discover_extr_delays)
        timers = timers_extr
        sampled_case_starting_times = sampled_cases_val
        start_timestamp = sampled_case_starting_times[0]
        sampled_case_starting_times = sampled_case_starting_times[1:]

        # Create the model using the loaded data
        business_process_model = BusinessProcessModel(business_process_data, activity_durations_dict, 
                                                        sampled_case_starting_times, roles, res_calendars, start_timestamp,
                                                        agent_activity_mapping, transition_probabilities_autonomous, prerequisites, parallel_activities,
                                                        max_activity_count_per_case, parallels_probs_dict, timers, discover_parallel_work, 
                                                        multitasking_probs_per_resource, max_multitasking_activities, activities_without_waiting_time,
                                                        agent_transition_probabilities_autonomous, central_orchestration, agentminer_nets, lstm_model)

        # define list of cases
        case_id = 0
        case_ = Case(case_id=case_id, start_timestamp=start_timestamp, agentminer_nets=agentminer_nets) # first case
        cases = [case_]

        # Run the model for a specified number of steps
        while business_process_model.sampled_case_starting_times: # while cases list is not empty
            business_process_model.step(cases)
        simulated_log_val_extr_autonomous = pd.DataFrame(STEPS_TAKEN)
        STEPS_TAKEN = [] # reset

        # 4) simulate val log without extr delays and autonomous handover
        discover_extr_delays = False
        central_orchestration = False
        timers = get_times_for_extrt_delays(discover_extr_delays)
        sampled_case_starting_times = sampled_cases_val
        start_timestamp = sampled_case_starting_times[0]
        sampled_case_starting_times = sampled_case_starting_times[1:]

        # Create the model using the loaded data
        business_process_model = BusinessProcessModel(business_process_data, activity_durations_dict, 
                                                        sampled_case_starting_times, roles, res_calendars, start_timestamp,
                                                        agent_activity_mapping, transition_probabilities_autonomous, prerequisites, parallel_activities,
                                                        max_activity_count_per_case, parallels_probs_dict, timers, discover_parallel_work, 
                                                        multitasking_probs_per_resource, max_multitasking_activities, activities_without_waiting_time,
                                                        agent_transition_probabilities_autonomous, central_orchestration, agentminer_nets, lstm_model)

        # define list of cases
        case_id = 0
        case_ = Case(case_id=case_id, start_timestamp=start_timestamp, agentminer_nets=agentminer_nets) # first case
        cases = [case_]

        # Run the model for a specified number of steps
        while business_process_model.sampled_case_starting_times: # while cases list is not empty
            business_process_model.step(cases)
        simulated_log_val_autonomous = pd.DataFrame(STEPS_TAKEN)
        # print(f"number of simulated cases: {len(business_process_model.past_cases)}")
        STEPS_TAKEN = [] # reset

        # 5) compute cycle time and check which one is closer to the val log
        from log_distance_measures.config import EventLogIDs
        from log_distance_measures.cycle_time_distribution import cycle_time_distribution_distance

        # Set event log column ID mapping
        event_log_ids = EventLogIDs(  # These values are stored in DEFAULT_CSV_IDS
            case="case_id",
            activity="activity_name",
            start_time="start_timestamp",
            end_time="end_timestamp",
            resource='resource'
        )

        ctdd_extr = cycle_time_distribution_distance(
                    df_val, event_log_ids,  # First event log and its column id mappings
                    simulated_log_val_extr, event_log_ids,  # Second event log and its column id mappings
                    bin_size=pd.Timedelta(hours=1)  # Bins of 1 hour
                )
        
        ctdd = cycle_time_distribution_distance(
                    df_val, event_log_ids,  # First event log and its column id mappings
                    simulated_log_val_, event_log_ids,  # Second event log and its column id mappings
                    bin_size=pd.Timedelta(hours=1)  # Bins of 1 hour
                )
        
        ctdd_extr_autonomous = cycle_time_distribution_distance(
                    df_val, event_log_ids,  # First event log and its column id mappings
                    simulated_log_val_extr_autonomous, event_log_ids,  # Second event log and its column id mappings
                    bin_size=pd.Timedelta(hours=1)  # Bins of 1 hour
                )
        
        ctdd_autonomous = cycle_time_distribution_distance(
                    df_val, event_log_ids,  # First event log and its column id mappings
                    simulated_log_val_autonomous, event_log_ids,  # Second event log and its column id mappings
                    bin_size=pd.Timedelta(hours=1)  # Bins of 1 hour
                )
        print(f"CTD extr + central: {ctdd_extr}")
        print(f"CTD without extr + central: {ctdd}")
        print(f"CTD extr + autonomous: {ctdd_extr_autonomous}")
        print(f"CTD without extr + autonomous: {ctdd_autonomous}")

        # 4) set the hyperparameter for extr_delays and the architecture
        ct_results_val = [ctdd_extr,ctdd,ctdd_extr_autonomous,ctdd_autonomous]
        best_val_result = min(ct_results_val)

        if ctdd_extr == best_val_result:
            timers = timers_extr
            discover_delays = True
            central_orchestration = True
        elif ctdd == best_val_result:
            timers = timers
            discover_delays = False
            central_orchestration = True
        elif ctdd_extr_autonomous == best_val_result:
            timers = timers_extr
            discover_delays = True
            central_orchestration = False
            transition_probabilities = transition_probabilities_autonomous
            agent_transition_probabilities = agent_transition_probabilities_autonomous
        elif ctdd_autonomous == best_val_result:
            timers = timers
            discover_delays = False
            central_orchestration = False
            transition_probabilities = transition_probabilities_autonomous
            agent_transition_probabilities = agent_transition_probabilities_autonomous         
    else:
        if discover_delays:
            timers = timers_extr
        else:
            timers = timers

        if central_orchestration == False:
            transition_probabilities = transition_probabilities_autonomous
            agent_transition_probabilities = agent_transition_probabilities_autonomous


    print(f"discover extr. delays: {discover_delays}")
    print(f"central orchestration: {central_orchestration}")


    # 6) simulate 10 times the test log
    for i in range(10):
        sampled_case_starting_times = sampled_cases
        start_timestamp = sampled_case_starting_times[0]
        sampled_case_starting_times = sampled_case_starting_times[1:]

        # Create the model using the loaded data
        business_process_model = BusinessProcessModel(business_process_data, activity_durations_dict, 
                                                      sampled_case_starting_times, roles, res_calendars, start_timestamp,
                                                      agent_activity_mapping, transition_probabilities, prerequisites, parallel_activities,
                                                      max_activity_count_per_case, parallels_probs_dict, timers, discover_parallel_work, 
                                                      multitasking_probs_per_resource, max_multitasking_activities, activities_without_waiting_time,
                                                      agent_transition_probabilities, central_orchestration, agentminer_nets, lstm_model)

        # define list of cases
        case_id = 0
        case_ = Case(case_id=case_id, start_timestamp=start_timestamp, agentminer_nets=agentminer_nets) # first case
        cases = [case_]

        # Run the model for a specified number of steps
        while business_process_model.sampled_case_starting_times: # while cases list is not empty
            business_process_model.step(cases)
            
        print(f"number of simulated cases: {len(business_process_model.past_cases)}")


        # Record steps taken by each agent to a single CSV file
        simulated_log = pd.DataFrame(STEPS_TAKEN)
        # add resource column
        simulated_log['resource'] = simulated_log['agent'].map(AGENT_TO_RESOURCE)
        # save log to csv
        path_to_file = os.path.join(data_dir,f"simulated_log_{i}.csv")
        simulated_log.to_csv(path_to_file, index=False)
        print(f"Simulated logs are stored in {path_to_file}")

        # reset STEPS_TAKEN
        STEPS_TAKEN = []
    print(f"configuration setting: orchestrated handovers: {central_orchestration}, extr. delays: {discover_delays}")