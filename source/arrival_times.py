import pandas as pd
import numpy as np
import scipy.stats as st
import math
from source.arrival_distribution import get_best_fitting_distribution, get_inter_arrival_times, get_average_distribution

def get_case_arrival_times(df, start_timestamp, num_cases_to_simulate, train=True, train_params=None):
    if train:
        arrival_distribution, min_max_time_per_day, average_occurrences_by_day = get_arrival_parameters_for_train(df)
        train_params = (arrival_distribution, min_max_time_per_day, average_occurrences_by_day)
    else:
        arrival_distribution, min_max_time_per_day, average_occurrences_by_day = train_params

    date_of_current_timestamp = start_timestamp
    day_of_current_timestamp = date_of_current_timestamp.strftime('%A').upper()
    sampled_cases = []

    first_day = True
    while len(sampled_cases) <= num_cases_to_simulate:
        date_string = date_of_current_timestamp.strftime('%Y-%m-%d')
        if day_of_current_timestamp in min_max_time_per_day.keys():
            min_timestamp = min_max_time_per_day[day_of_current_timestamp][0].time().strftime('%H:%M:%S')
            max_timestamp = min_max_time_per_day[day_of_current_timestamp][1].time().strftime('%H:%M:%S')
            x = round(average_occurrences_by_day[day_of_current_timestamp][0])

            times = random_sample_timestamps_(date_string, min_timestamp, max_timestamp, x, 
                                              arrival_distribution, first_day, start_timestamp)
            first_day = False
            sampled_cases.extend(times)
            day_of_current_timestamp = increment_day_of_week(day_of_current_timestamp)
            date_of_current_timestamp += pd.Timedelta(days=1)
        
        else:
            day_of_current_timestamp = increment_day_of_week(day_of_current_timestamp)
            date_of_current_timestamp += pd.Timedelta(days=1)

    sampled_cases = sampled_cases[:num_cases_to_simulate + 1]

    return sampled_cases, train_params

def get_arrival_parameters_for_train(df):
    arrival_distribution = _get_arrival_distribution(df)
    case_start_timestamps = df.groupby('case_id')['start_timestamp'].min().tolist()
    min_max_time_per_day = get_min_max_time_per_day(case_start_timestamps)
    average_occurrences_by_day = get_average_occurence_of_cases_per_day(case_start_timestamps)

    return arrival_distribution, min_max_time_per_day, average_occurrences_by_day

def _get_arrival_distribution(df_train):
    inter_arrival_durations = get_inter_arrival_times(df_train)
    # arrival_distribution = get_best_fitting_distribution(
    #     data=inter_arrival_durations,
    #     filter_outliers=False,
    #     outlier_threshold=20.0,
    # )
    arrival_distribution = get_average_distribution(
        data=inter_arrival_durations,
        filter_outliers=False,
        outlier_threshold=20.0,
    )

    return arrival_distribution

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