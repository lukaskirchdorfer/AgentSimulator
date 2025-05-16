from typing import List, Optional
import pandas as pd

from source.agent_types.resource_calendar import RCalendar
from source.agent_types.calendar_discovery_parameters import CalendarDiscoveryParameters, CalendarType
from source.agent_types.factory import CalendarFactory
from source.agent_types.roles import ResourceProfile

def _update_resource_calendars(resource_profiles: List[ResourceProfile], calendar_id: str):
    for resource_profile in resource_profiles:
        for resource in resource_profile.resources:
            resource.calendar_id = calendar_id

def discover_resource_calendars_per_profile(
    event_log: pd.DataFrame,
    # log_ids: EventLogIDs,
    params: CalendarDiscoveryParameters,
    resource_profiles: List[ResourceProfile],
    pools,
) -> List[RCalendar]:
    # Revert resource profiles
    resource_to_profile = {
        resource.id: resource_profile.id
        for resource_profile in resource_profiles
        for resource in resource_profile.resources
    }

    # --- Discover a calendar per resource profile --- #

    # Register each timestamp to its corresponding profile
    calendar_factory = CalendarFactory(params.granularity)
    for _, event in event_log.iterrows():
        # Register start/end timestamps
        profile_id = resource_to_profile[event['agent']]
        activity = event['activity_name']
        calendar_factory.check_date_time(profile_id, activity, event['start_timestamp'])
        calendar_factory.check_date_time(profile_id, activity, event['end_timestamp'])

    # Discover weekly timetables
    discovered_timetables = calendar_factory.build_weekly_calendars(
        params.confidence, params.support, params.participation
    )

    pool_names = list(pools.keys())
    resource_profile_names = list(discovered_timetables.keys())
    missing_profiles = []
    for i in range(len(pool_names)):
        for j in range(len(resource_profile_names)):
            if pool_names[i] in resource_profile_names[j]:
                if discovered_timetables[resource_profile_names[j]] is not None:
                    pools[pool_names[i]]['calendar'] = discovered_timetables[resource_profile_names[j]].to_dict()['time_periods']
                else:
                    if resource_profile_names[j] not in missing_profiles:
                        missing_profiles.append(resource_profile_names[j])

    missing_agents = []
    for profile in missing_profiles:
        for key, value in pools.items():
            if profile[:-8] == key:
                missing_agents.append(value)
    missing_agents = [element for sublist in missing_agents for element in sublist]


    if len(missing_profiles) > 0:
        filtered_event_log = event_log[event_log['agent'].isin(missing_agents)]
        missing_resource_calendar = _create_full_day_calendar()

        # # Discover one resource calendar for all of them
        # # print("try to discover calendar with filtered event log")
        # missing_resource_calendar = _discover_undifferentiated_resource_calendar(filtered_event_log, params)
        # if missing_resource_calendar is None or len(missing_resource_calendar) == 0:
        #     # print("try to discover calendar with full event log")
        #     # Could not discover calendar for the missing resources, discover calendar with the entire log
        #     missing_resource_calendar = _discover_undifferentiated_resource_calendar(event_log, params)
        #     if missing_resource_calendar is None or len(missing_resource_calendar) == 0:
        #         # print("set 24/7 calendar")
        #         # Could not discover calendar for all the resources in the log, assign default 24/7
        #         missing_resource_calendar = _create_full_day_calendar()

        pool_names = list(pools.keys())
        resource_profile_names = list(missing_resource_calendar.keys())
        # print(f"pool names: {pool_names}")
        # print(f"resource profile names: {resource_profile_names}")
        # missing_profiles = []
        for i in range(len(pool_names)):
            for j in range(len(missing_profiles)):
                if pool_names[i] == missing_profiles[j][:-8]:
                    if missing_resource_calendar['Undifferentiated'] is not None:
                        pools[pool_names[i]]['calendar'] = missing_resource_calendar['Undifferentiated'].to_dict()['time_periods']
                    else:
                        if resource_profile_names[j] not in missing_profiles:
                            missing_profiles.append(resource_profile_names[j])
        # print(f"missing profiles: {missing_profiles}")

    # Create calendar per resource profile
    # resource_calendars = []
    # missing_profiles = []
    # for resource_profile in resource_profiles:
    #     calendar_id = f"{resource_profile.id}_calendar"
    #     discovered_calendar = discovered_timetables.get(resource_profile.id)
    #     if (discovered_calendar is not None) and (not discovered_calendar.is_empty()):
    #         discovered_calendar.calendar_id = calendar_id
    #         resource_calendars += [discovered_calendar]
    #         _update_resource_calendars([resource_profile], calendar_id)
    #     else:
    #         missing_profiles += [resource_profile]

    # Check if there are resources with no calendars assigned
    # if len(missing_profiles) > 0:
    #     # Retain events performed by the resources with no calendar
    #     missing_resource_ids = [
    #         resource.id for resource_profile in resource_profiles for resource in resource_profile.resources
    #     ]
    #     filtered_event_log = event_log[event_log[log_ids.resource].isin(missing_resource_ids)]
    #     # Discover one resource calendar for all of them
    #     missing_resource_calendar = _discover_undifferentiated_resource_calendar(filtered_event_log, params)
    #     if missing_resource_calendar is None:
    #         # Could not discover calendar for the missing resources, discover calendar with the entire log
    #         missing_resource_calendar = _discover_undifferentiated_resource_calendar(event_log, log_ids, params)
    #         if missing_resource_calendar is None:
    #             # Could not discover calendar for all the resources in the log, assign default 24/7
    #             missing_resource_calendar = _create_full_day_calendar()
    #     # Add grouped calendar to discovered resource calendars
    #     resource_calendars += [missing_resource_calendar]
    #     # Set common calendar id to missing resources
    #     _update_resource_calendars(missing_profiles, missing_resource_calendar.calendar_id)
    # # Return resource calendars
    resource_calendars = None
    return resource_calendars, pools


def _discover_undifferentiated_resource_calendar(
    event_log: pd.DataFrame,
    # log_ids: EventLogIDs,
    params: CalendarDiscoveryParameters,
    # calendar_id: str = "Undifferentiated_calendar",
) -> Optional[RCalendar]:
    """
    Discover one availability calendar using all the timestamps in the received event log.

    :param event_log: event log to discover the resource calendar from.
    :param log_ids: column IDs of the event log.
    :param params: parameters for the calendar discovery.
    :param calendar_id: ID to assign to the discovered calendar.

    :return: resource calendar for all the events in the received event log.
    """
    # Register each timestamp to the same profile
    calendar_factory = CalendarFactory(params.granularity)
    for _, event in event_log.iterrows():
        # Register start/end timestamps
        activity = event['activity_name']
        calendar_factory.check_date_time("Undifferentiated", activity, event['start_timestamp'])
        calendar_factory.check_date_time("Undifferentiated", activity, event['end_timestamp'])
    # Discover weekly timetables
    discovered_timetables = calendar_factory.build_weekly_calendars(
        params.confidence, params.support, params.participation
    )
    # # Get discovered calendar and update ID if discovered
    # undifferentiated_calendar = discovered_timetables.get("Undifferentiated")
    # if undifferentiated_calendar is not None:
    #     undifferentiated_calendar.calendar_id = calendar_id
    # Return resource calendar
    return discovered_timetables

def _create_full_day_calendar(schedule_id: str = "24_7_CALENDAR") -> RCalendar:
    schedule = RCalendar(schedule_id)
    schedule.add_calendar_item(
        from_day="MONDAY",
        to_day="SUNDAY",
        begin_time="00:00:00.000",
        end_time="23:59:59.999",
    )
    return schedule