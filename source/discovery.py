import pandas as pd
import numpy as np
import json
from source.utils import store_preprocessed_data
from source.agent_types.discover_roles import discover_roles_and_calendars
from source.agent_types.discover_resource_calendar import discover_calendar_per_agent
from source.arrival_distribution import get_best_fitting_distribution
from source.arrival_times import get_case_arrival_times
from source.activity_transition import compute_activity_transition_dict, compute_activity_transition_dict_global
from source.interaction_probabilities import calculate_agent_handover_probabilities_per_activity
from source.extraneous_delays.config import (
    Configuration as ExtraneousActivityDelaysConfiguration,
    TimerPlacement,
)
from source.extraneous_delays.delay_discoverer import compute_complex_extraneous_activity_delays, compute_naive_extraneous_activity_delays
from source.extraneous_delays.event_log import EventLogIDs
from source.simulation import BusinessProcessModel, Case

def discover_simulation_parameters(df_train, df_test, df_val, data_dir, num_cases_to_simulate, num_cases_to_simulate_val, determine_automatically=False, central_orchestration=False, discover_extr_delays=False):
    """
    Discover the simulation model from the training data.
    """
    df_train, agent_to_resource = preprocess(df_train)
    df_test, _ = preprocess(df_test)
    df_val, _ = preprocess(df_val)
    START_TIME = min(df_train.groupby('case_id')['start_timestamp'].min().to_list())
    START_TIME_VAL = min(df_val.groupby('case_id')['start_timestamp'].min().to_list())


    df_train_without_end_activity = store_preprocessed_data(df_train, df_test, df_val, data_dir)

    activities_without_waiting_time = activities_with_zero_waiting_time(df_train)

    # extract roles and calendars  
    roles = discover_roles_and_calendars(df_train_without_end_activity)
    res_calendars, _, _, _, _ = discover_calendar_per_agent(df_train_without_end_activity)

    activity_durations_dict = compute_activity_duration_distribution_per_agent(df_train, res_calendars, roles)

    # define mapping of agents to activities based on event log
    agent_activity_mapping = df_train.groupby('agent')['activity_name'].unique().apply(list).to_dict()

    transition_probabilities_autonomous = compute_activity_transition_dict(df_train)
    agent_transition_probabilities_autonomous = calculate_agent_handover_probabilities_per_activity(df_train)
    agent_transition_probabilities = None
    transition_probabilities = compute_activity_transition_dict_global(df_train)

    prerequisites, parallel_activities = get_prerequisites_per_activity(df_train)

    # get maximum activity frequency per case
    activity_counts = df_train.groupby(['case_id', 'activity_name']).size().reset_index(name='count')
    max_activity_count_per_case = activity_counts.groupby('activity_name')['count'].max().to_dict()


    # sample arrival times for training and validation data
    # case_arrival_times, train_params = get_case_arrival_times(df_train, start_timestamp=START_TIME, num_cases_to_simulate=num_cases_to_simulate, train=True)

    with open('arrival_times/ATKDE/Confidential_1000.json', 'r') as f:
        arrivals = json.load(f)

    case_arrival_times = []
    for arr in arrivals:
        # Transform string timestamp to pandas Timestamp
        timestamp = pd.to_datetime(arr, format='%d.%m.%Y %H:%M:%S', utc=True)
        case_arrival_times.append(timestamp)

    # case_arrival_times_val, _ = get_case_arrival_times(df_val, start_timestamp=START_TIME_VAL, num_cases_to_simulate=num_cases_to_simulate_val, train=False, train_params=train_params)
    case_arrival_times_val = None

    simulation_parameters = {
        'activity_durations_dict': activity_durations_dict,
        'activities_without_waiting_time': activities_without_waiting_time,
        'roles': roles,
        'res_calendars': res_calendars,
        'agent_activity_mapping': agent_activity_mapping,
        'transition_probabilities_autonomous': transition_probabilities_autonomous,
        'agent_transition_probabilities_autonomous': agent_transition_probabilities_autonomous,
        'agent_transition_probabilities': agent_transition_probabilities,
        'transition_probabilities': transition_probabilities,
        'max_activity_count_per_case': max_activity_count_per_case,
        'case_arrival_times': case_arrival_times,
        # 'case_arrival_times_val': case_arrival_times_val,
        'agent_to_resource': agent_to_resource,
        'determine_automatically': determine_automatically,
        'prerequisites': prerequisites,
    }

    simulation_parameters = determine_agent_behavior_type_and_extraneous_delays(simulation_parameters, df_train, df_val, case_arrival_times_val, central_orchestration, discover_extr_delays)
    simulation_parameters['start_timestamp'] = START_TIME


    return df_train, simulation_parameters

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

    # Apply the function to each row
    df['resource'] = df.apply(rename_artificial, axis=1)
    # name agents with plain integers 
    df['agent'] = pd.factorize(df['resource'])[0]
    # Create a mapping of integers to resource values
    integers = df['agent'].unique()
    resources = df['resource'].unique()
    integers_to_resources = dict(zip(integers, resources))
    # Create the agent_to_resource mapping dictionary by reversing the integers_to_resources dictionary
    agent_to_resource = {k: v for k, v in integers_to_resources.items()}
    # insert a new row after every ending case
    df = insert_rows_before_case_change(df)
    df['start_timestamp']= pd.to_datetime(df['start_timestamp'], utc=True, format='mixed')
    df['end_timestamp']= pd.to_datetime(df['end_timestamp'], utc=True, format='mixed')

    return df, agent_to_resource


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

def activities_with_zero_waiting_time(df, threshold=0.99):
    """
    Returns a list of activities that have zero waiting time in the log.
    """
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

def get_prerequisites_per_activity(data):
    preceding_activities_dict = generate_preceding_activities_dict(data)
    print(f"preceding_activities_dict: {preceding_activities_dict}")
    # x=y
    return preceding_activities_dict, None

# def get_prerequisites_per_activity(data, discover_parallel_work=True):
#     # Check for parallel activities
#     result = []
#     # Group by case_id
#     grouped = data.groupby('case_id')
#     # Iterate over groups
#     for case_id, group in grouped:
#         # Call the function to check for parallel activities with a minimum of 2
#         parallel_activities = check_parallel_activities(group, min_activities=2)
        
#         # Extend the result list with the detected parallel activities
#         result.extend([(case_id,) + tuple(activities) for activities in parallel_activities])

#     parallel_activities = get_unique_parallel_activities(result=result)
#     # print(f"parallel_activities: {parallel_activities}")

#     preceding_activities_dict = generate_preceding_activities_dict(data)
#     # print(f"preceeding activties: {preceding_activities_dict}")
#     # remove parallel activities as prerequisite for each other
#     for i in range(len(parallel_activities)):
#         for key, value in preceding_activities_dict.items():
#             if key in parallel_activities[i]:
#                 par = parallel_activities[i]
#                 related_activities = [item for item in par if item != key]
#                 for j in related_activities:
#                     if j in value: # TODO understand why we need to add this
#                         value.remove(j)
#             # join parallel activities in prerequisite of other activities to mark that they are both required
#             parallels = parallel_activities[i]
#             value_flattened = [item for sublist in value for item in sublist]

#             if set(parallels).issubset(set(value_flattened)):
#                 # Check if 'parallels' is a subset of 'value_flattened'
#                 disjoint_part = list(set(parallels).symmetric_difference(set(value)))

#                 # Convert 'parallels' to a tuple before using it as a key
#                 parallels_tuple = tuple(parallels)

#                 # Use the tuple as a key in the dictionary
#                 preceding_activities_dict[key] = disjoint_part + [parallels_tuple]

#     # remove the activity itself as prerequisite for itself
#     new_dict = {key: [] for key, value in preceding_activities_dict.items()}
#     for key, value in preceding_activities_dict.items():     
#         for i in range(len(value)):
#             if not isinstance(value[i], list):
#                 if not value[i] == key:
#                     new_dict[key].append(value[i])

#             # if value contains sublists
#             else:
#                 if key in value[i]:
#                     value[i].remove(key)
#                     new_dict[key].append(value[i])
#                 else:
#                     new_dict[key].append(value[i])

#     preceding_activities_dict = new_dict

#     if discover_parallel_work == False:
#         parallel_activities = []
#     # print(f"preceding_activities_dict: {preceding_activities_dict}")

#     return preceding_activities_dict, parallel_activities
    

def check_parallel_activities(group, min_activities=2):
    sorted_group = group.sort_values(by='start_timestamp')
    result = []

    # Iterate over the range of parallel activities (2 or more)
    for i in range(len(sorted_group) - min_activities + 1):
        current_end_time = sorted_group.iloc[i]['end_timestamp']
        if sorted_group.iloc[i]['end_timestamp'] - sorted_group.iloc[i]['start_timestamp'] > pd.Timedelta(seconds=0):
            # print(f"current activity: {sorted_group.iloc[i]['activity_name']}")
            parallel_activities = [sorted_group.iloc[j]['activity_name'] for j in range(i + 1, i + min_activities) if current_end_time > sorted_group.iloc[j]['start_timestamp'] and sorted_group.iloc[j]['end_timestamp'] - sorted_group.iloc[j]['start_timestamp'] > pd.Timedelta(seconds=0)]
            if len(parallel_activities) == min_activities - 1:
                # print(f"parallel_activities: {parallel_activities}")
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

def intersect_lists(list_of_lists):
    """
    Returns the intersection of all lists in a list of lists.
    
    Args:
        list_of_lists (list of lists): A list containing multiple lists.
    
    Returns:
        list: A list containing elements common to all input lists.
    """
    if not list_of_lists:
        return []
    
    # Use set intersection to find common elements
    intersection = set(list_of_lists[0])
    for lst in list_of_lists[1:]:
        intersection &= set(lst)
    
    # Return the intersection as a list 
    intersection = list(intersection)

    return intersection


def generate_preceding_activities_dict(data):
    preceding_activities_dict = {}
    preceding_activities_dict_clean = {}

    # Group by case_id
    grouped = data.groupby('case_id')

    # Iterate over groups
    for case_id, group in grouped:
        sorted_group = group.sort_values(by='end_timestamp')

        for i in range(1, len(sorted_group)):
            current_activity = sorted_group.iloc[i]['activity_name']
            # print(f"current_activity: {current_activity}")
            preceding_activities = [sorted_group.iloc[i - j]['activity_name'] for j in range(1, i+1)]
            # print(f"preceding_activities: {preceding_activities}")

            if current_activity not in preceding_activities_dict:
                preceding_activities_dict[current_activity] = []
            preceding_activities_dict[current_activity].append(preceding_activities)
        
    # print(f"preceding_activities_dict: {preceding_activities_dict}")

    # iterate over all keys and extract the activities that appear in each sublist of that key
    for key, value in preceding_activities_dict.items():
        preceding_activities_dict_clean[key] = intersect_lists(value)

    # print(f"preceding_activities_dict_clean: {preceding_activities_dict_clean}")

    return preceding_activities_dict_clean

# def generate_preceding_activities_dict(data):
#     preceding_activities_dict = {}

#     # Group by case_id
#     grouped = data.groupby('case_id')

#     # Iterate over groups
#     for case_id, group in grouped:
#         sorted_group = group.sort_values(by='end_timestamp')
#         print(f"first activity: {sorted_group.iloc[0]['activity_name']}")

#         # Iterate through the sorted activities
#         for i in range(1, len(sorted_group)):
#             current_activity = sorted_group.iloc[i]['activity_name']
#             preceding_activity = sorted_group.iloc[i - 1]['activity_name']

#             # Update the dictionary with the preceding activity
#             if current_activity not in preceding_activities_dict:
#                 preceding_activities_dict[current_activity] = set()
#             preceding_activities_dict[current_activity].add(preceding_activity)

#     # Convert sets to lists
#     preceding_activities_dict = {key: list(value) for key, value in preceding_activities_dict.items()}

#     print(f"preceding_activities_dict: {preceding_activities_dict}")
#     return preceding_activities_dict

def _get_times_for_extr_delays(df_train, discover_extr_delays=True):
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
                df_train,
                configuration,
                configuration.should_consider_timer,
                )
        elif discovery_method == "complex":
            timers = compute_complex_extraneous_activity_delays(
                df_train,
                configuration,
                configuration.should_consider_timer,
            )
    else:
        timers = {}

    return timers


def determine_agent_behavior_type_and_extraneous_delays(simulation_parameters, df_train, df_val, case_arrival_times_val, central_orchestration_parameter, discover_extr_delays_parameter):
    """
    Determine the agent behavior type and extraneous delays.
    """
    timers_extr = _get_times_for_extr_delays(df_train, discover_extr_delays=True)
    timers = _get_times_for_extr_delays(df_train, discover_extr_delays=False)
    # create a copy of the simulation parameters such that we can modify it without changing the original one
    simulation_parameters_copy = simulation_parameters.copy()
    
    if simulation_parameters_copy['determine_automatically']:
        # 1) simulate val log with extr delays and central orchestration
        central_orchestration = True
        sampled_case_starting_times = case_arrival_times_val
        start_timestamp = sampled_case_starting_times[0]
        sampled_case_starting_times = sampled_case_starting_times[1:]

        simulation_parameters_copy['sampled_case_starting_times'] = sampled_case_starting_times
        simulation_parameters_copy['start_timestamp'] = start_timestamp
        simulation_parameters_copy['timers'] = timers_extr
        simulation_parameters_copy['central_orchestration'] = central_orchestration
        # Create the model using the loaded data
        business_process_model = BusinessProcessModel(df_train, simulation_parameters_copy)
        # define list of cases
        case_id = 0
        case_ = Case(case_id=case_id, start_timestamp=start_timestamp) # first case
        cases = [case_]
        # Run the model for a specified number of steps
        while business_process_model.sampled_case_starting_times: # while cases list is not empty
            business_process_model.step(cases)
        simulated_log_val_extr = pd.DataFrame(business_process_model.simulated_events)

        # 2) simulate val log without extr delays and central orchestration
        simulation_parameters_copy['timers'] = timers
        sampled_case_starting_times = case_arrival_times_val
        simulation_parameters_copy['start_timestamp'] = sampled_case_starting_times[0]
        simulation_parameters_copy['sampled_case_starting_times'] = sampled_case_starting_times[1:]
        simulation_parameters_copy['central_orchestration'] = True
        # Create the model using the loaded data
        business_process_model = BusinessProcessModel(df_train, simulation_parameters_copy)
        # define list of cases
        case_id = 0
        case_ = Case(case_id=case_id, start_timestamp=start_timestamp) # first case
        cases = [case_]
        # Run the model for a specified number of steps
        while business_process_model.sampled_case_starting_times: # while cases list is not empty
            business_process_model.step(cases)
        simulated_log_val_ = pd.DataFrame(business_process_model.simulated_events)

        # 3) simulate val log with extr delays and autonomous handover
        simulation_parameters_copy['central_orchestration'] = False
        simulation_parameters_copy['timers'] = timers_extr
        sampled_case_starting_times = case_arrival_times_val
        simulation_parameters_copy['start_timestamp'] = sampled_case_starting_times[0]
        simulation_parameters_copy['sampled_case_starting_times'] = sampled_case_starting_times[1:]
        simulation_parameters_copy['transition_probabilities'] = simulation_parameters_copy['transition_probabilities_autonomous']
        simulation_parameters_copy['agent_transition_probabilities'] = simulation_parameters_copy['agent_transition_probabilities_autonomous']
        # Create the model using the loaded data
        business_process_model = BusinessProcessModel(df_train, simulation_parameters_copy)
        # define list of cases
        case_id = 0
        case_ = Case(case_id=case_id, start_timestamp=start_timestamp) # first case
        cases = [case_]
        # Run the model for a specified number of steps
        while business_process_model.sampled_case_starting_times: # while cases list is not empty
            business_process_model.step(cases)
        simulated_log_val_extr_autonomous = pd.DataFrame(business_process_model.simulated_events)

        # 4) simulate val log without extr delays and autonomous handover
        simulation_parameters_copy['timers'] = timers
        sampled_case_starting_times = case_arrival_times_val
        simulation_parameters_copy['start_timestamp'] = sampled_case_starting_times[0]
        simulation_parameters_copy['sampled_case_starting_times'] = sampled_case_starting_times[1:]
        simulation_parameters_copy['central_orchestration'] = False
        simulation_parameters_copy['transition_probabilities'] = simulation_parameters_copy['transition_probabilities_autonomous']
        simulation_parameters_copy['agent_transition_probabilities'] = simulation_parameters_copy['agent_transition_probabilities_autonomous']
        # Create the model using the loaded data
        business_process_model = BusinessProcessModel(df_train, simulation_parameters_copy)
        # define list of cases
        case_id = 0
        case_ = Case(case_id=case_id, start_timestamp=start_timestamp) # first case
        cases = [case_]
        # Run the model for a specified number of steps
        while business_process_model.sampled_case_starting_times: # while cases list is not empty
            business_process_model.step(cases)
        simulated_log_val_autonomous = pd.DataFrame(business_process_model.simulated_events)

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
            simulation_parameters['timers'] = timers_extr
            discover_delays = True
            simulation_parameters['central_orchestration'] = True
        elif ctdd == best_val_result:
            simulation_parameters['timers'] = timers
            discover_delays = False
            simulation_parameters['central_orchestration'] = True
        elif ctdd_extr_autonomous == best_val_result:
            simulation_parameters['timers'] = timers_extr
            discover_delays = True
            simulation_parameters['central_orchestration'] = False
            simulation_parameters['transition_probabilities'] = simulation_parameters['transition_probabilities_autonomous']
            simulation_parameters['agent_transition_probabilities'] = simulation_parameters['agent_transition_probabilities_autonomous']
        elif ctdd_autonomous == best_val_result:
            simulation_parameters['timers'] = timers
            discover_delays = False
            simulation_parameters['central_orchestration'] = False
            simulation_parameters['transition_probabilities'] = simulation_parameters['transition_probabilities_autonomous']
            simulation_parameters['agent_transition_probabilities'] = simulation_parameters['agent_transition_probabilities_autonomous']
    else:
        if discover_extr_delays_parameter == True:
            simulation_parameters['timers'] = timers_extr
        else:
            simulation_parameters['timers'] = timers

        simulation_parameters['central_orchestration'] = central_orchestration_parameter
        if central_orchestration_parameter == False:
            simulation_parameters['transition_probabilities'] = simulation_parameters['transition_probabilities_autonomous']
            simulation_parameters['agent_transition_probabilities'] = simulation_parameters['agent_transition_probabilities_autonomous']


    # print(f"discover extr. delays: {discover_delays}")
    # print(f"central orchestration: {central_orchestration}")

    return simulation_parameters