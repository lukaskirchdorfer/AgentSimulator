import pandas as pd
from datetime import datetime
import pytz
from mesa import Agent

import bisect


from source.utils import sample_from_distribution

class ResourceAgent(Agent):
    """
    One agent for each resource in the event log
    """
    def __init__(self, unique_id, model, resource, timer, contractor_agent=None):
        super().__init__(unique_id, model)
        self.resource = resource
        self.model = model
        self.is_busy = False
        self.is_busy_until = None
        self.contractor_agent = contractor_agent
        self.agent_type = next((role for role, ids in self.model.roles.items() if self.resource in ids['agents']), None)
        if self.resource in self.model.calendars.keys():
            self.calendar = self.model.calendars[self.resource].intervals_to_json()
        else:
            self.calendar = next((ids['calendar'] for role, ids in self.model.roles.items() if self.resource in ids['agents']), None)
        self.timer = timer
        self.occupied_times = []

    def step(self, last_possible_agent=False, parallel_activity=False, current_timestamp=None, perform_multitask=False):
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
            waiting_time_distribution = self.timer[activity]
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
                    self.add_occupied_slot(current_timestamp, current_timestamp + pd.Timedelta(seconds=activity_duration))

                    self.is_busy_until = current_timestamp + pd.Timedelta(seconds=activity_duration)
                    self.is_busy = True
                    self.model.agents_busy_until[self.resource] = self.is_busy_until
                else:
                    pass
                # advance current timestamp
                self.contractor_agent.case.current_timestamp = current_timestamp + pd.Timedelta(seconds=activity_duration)
                # add activity to case list to keep track of performed activities per case
                self.contractor_agent.case.add_activity_to_case(activity)
                # print(f"Activity performed: {activity}")

                # set that activity is performed
                self.contractor_agent.activity_performed = True

                self.contractor_agent.case.previous_agent = self.resource

    
                # remove activity from additional activities
                if additional_act == True:
                    index_to_delete = self.contractor_agent.case.additional_next_activities.index(activity)
                    self.contractor_agent.case.additional_next_activities.pop(index_to_delete)


                self.model.simulated_events.append({'case_id': self.contractor_agent.case.case_id, 
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
                self.add_occupied_slot(current_timestamp, end_time)
                self.is_busy_until = end_time
                self.is_busy = True
                self.model.agents_busy_until[self.resource] = self.is_busy_until

                self.contractor_agent.case.current_timestamp = end_time
                self.contractor_agent.case.add_activity_to_case(activity)
                self.contractor_agent.activity_performed = True
                self.contractor_agent.case.previous_agent = self.resource

                self.model.simulated_events.append({'case_id': self.contractor_agent.case.case_id, 
                                    'agent': self.resource, 
                                    'activity_name': activity,
                                    'start_timestamp': current_timestamp,
                                    'end_timestamp': self.contractor_agent.case.current_timestamp,
                                    'TimeStep': self.model.schedule.steps,
                                    })

            else:
                # print(f"#######agent {self.resource} is free but time not within calendar")
                # print(f"activity duration: {activity_duration}")
                if last_possible_agent: # then increase timer by x seconds to try to get an available agent later
                    # move timestamp until agent is available again according to calendar
                    self.contractor_agent.case.current_timestamp = self.set_time_to_next_availability_when_not_in_calendar(current_timestamp, activity_duration)
                    # print(f"set timestamp until agent is available again: {self.contractor_agent.case.current_timestamp}")
                    if additional_act == True:
                        self.model.additional_activity_index += 1
                else:
                    pass # first try if one of the other possible agents is available
        else:
            # print(f"agent {self.resource} is busy when trying to perform task {activity} until {self.is_busy_until}")
            if last_possible_agent: # then increase timer by x seconds to try to get an available agent later
                self.set_current_time_to_next_available_slot()
                if additional_act == True:
                    self.model.additional_activity_index += 1
            else:
                pass # first try if one of the other possible agents is available


    def is_occupied(self, new_start, activity_duration):
        if not self.occupied_times:
            return False

        new_end = new_start + pd.Timedelta(seconds=activity_duration)

        # Find where this new_start would sit in the sorted list
        idx = bisect.bisect_left(self.occupied_times, (new_start, new_end))

        # Check if the PREVIOUS slot ends after our new task starts
        if idx > 0 and new_start < self.occupied_times[idx - 1][1]:
            return True

        # Check if the NEXT slot starts before our new task ends
        if idx < len(self.occupied_times) and new_end > self.occupied_times[idx][0]:
            return True
        return False

    def get_current_number_multitasking(self, new_start, activity_duration):
        new_end = new_start + pd.Timedelta(seconds=activity_duration)
        number_multitask = 1  # 1 because we have to add the current activity as well
        for start, end in self.occupied_times:
            if new_start < end and new_end > start:
                number_multitask += 1

        return number_multitask

    def add_occupied_slot(self, start, end):
        # "Wrapper helper" to just insert a slot in the right spot
        bisect.insort(self.occupied_times, (start, end))

    def set_current_time_to_next_available_slot(self):
        # Updated func to store already sorted and remove old times
        current_time = self.contractor_agent.case.current_timestamp

        # Prune 'past' slots to keep the list short 
        self.occupied_times = [s for s in self.occupied_times if s[1] > current_time]

        if self.occupied_times:
            # Since it's sorted by start, the first remaining slot
            # is the most likely candidate for the 'next' end.
            self.contractor_agent.case.current_timestamp = self.occupied_times[0][1]
        else:
            self.contractor_agent.case.current_timestamp += pd.Timedelta(seconds=60)

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