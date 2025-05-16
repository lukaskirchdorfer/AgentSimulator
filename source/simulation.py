import pandas as pd
from mesa import Agent, Model
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector

from source.agents.contractor import ContractorAgent
from source.agents.resource import ResourceAgent
from source.utils import store_simulated_log


def simulate_process(df_train, simulation_parameters, data_dir, num_simulations):
    start_timestamp = simulation_parameters['case_arrival_times'][0]
    simulation_parameters['start_timestamp'] = start_timestamp
    simulation_parameters['case_arrival_times'] = simulation_parameters['case_arrival_times'][1:]
    for i in range(1):
        # Create the model using the loaded data
        business_process_model = BusinessProcessModel(df_train, simulation_parameters)

        # define list of cases
        case_id = 0
        case_ = Case(case_id=case_id, start_timestamp=start_timestamp) # first case
        cases = [case_]

        # Run the model for a specified number of steps
        while business_process_model.sampled_case_starting_times: # while cases list is not empty
            business_process_model.step(cases)
            
        print(f"number of simulated cases: {len(business_process_model.past_cases)}")

        # Record steps taken by each agent to a single CSV file
        simulated_log = pd.DataFrame(business_process_model.simulated_events)
        # add resource column
        simulated_log['resource'] = simulated_log['agent'].map(simulation_parameters['agent_to_resource'])
        # save log to csv
        store_simulated_log(data_dir, simulated_log, i)

class Case:
    """
    represents a case, for example a patient in a medical surveillance process
    """
    def __init__(self, case_id, start_timestamp=None, ) -> None:
        self.case_id = case_id
        self.is_done = False
        self.activities_performed = []
        self.case_start_timestamp = start_timestamp
        self.current_timestamp = start_timestamp
        self.additional_next_activities = []
        self.potential_additional_agents = []
        self.timestamp_before_and_gateway = start_timestamp
        self.previous_agent = -1

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
    
    def update_current_timestep(self, duration):
        self.current_timestamp += pd.Timedelta(seconds=duration)


class BusinessProcessModel(Model):
    def __init__(self, data, simulation_parameters):
        self.data = data
        self.resources = sorted(set(self.data['agent']))
        activities = sorted(set(self.data['activity_name']))

        self.roles = simulation_parameters['roles']
        self.agents_busy_until = {key: simulation_parameters['start_timestamp'] for key in self.resources}
        self.calendars = simulation_parameters['res_calendars']
        self.activity_durations_dict = simulation_parameters['activity_durations_dict']
        self.sampled_case_starting_times = simulation_parameters['case_arrival_times']
        self.past_cases = []
        self.maximum_case_id = 0
        self.prerequisites = simulation_parameters['prerequisites']
        self.max_activity_count_per_case = simulation_parameters['max_activity_count_per_case']
        self.timer = simulation_parameters['timers']
        self.activities_without_waiting_time = simulation_parameters['activities_without_waiting_time']
        self.agent_transition_probabilities = simulation_parameters['agent_transition_probabilities']
        self.central_orchestration = simulation_parameters['central_orchestration']
        self.discover_parallel_work = False
        self.schedule = MyScheduler(self,)
        self.contractor_agent = ContractorAgent(unique_id=9999, 
                                                model=self, 
                                                activities=activities, 
                                                transition_probabilities=simulation_parameters['transition_probabilities'], 
                                                agent_activity_mapping=simulation_parameters['agent_activity_mapping'])
        self.schedule.add(self.contractor_agent)

        for agent_id in range(len(self.resources)):
            agent = ResourceAgent(agent_id, self, self.resources[agent_id], self.timer, self.contractor_agent)
            self.schedule.add(agent)

        # Data collector to track agent activities over time
        self.datacollector = DataCollector(agent_reporters={"Activity": "current_activity_index"})
        self.simulated_events = []


    def step(self, cases):
        # check if there are still cases planned to arrive in the future
        if len(self.sampled_case_starting_times) > 1:
        # if there are still cases happening
            if cases:
                last_case = cases[-1]
                if last_case.current_timestamp >= self.sampled_case_starting_times[0]:
                    self.maximum_case_id += 1
                    new_case_id = self.maximum_case_id
                    new_case = Case(case_id=new_case_id, start_timestamp=self.sampled_case_starting_times[0])
                    cases.append(new_case)
                    # remove added case from sampled_case_starting_times list
                    self.sampled_case_starting_times = self.sampled_case_starting_times[1:]
            # if no cases are happening
            else:
                self.maximum_case_id += 1
                new_case_id = self.maximum_case_id
                new_case = Case(case_id=new_case_id, start_timestamp=self.sampled_case_starting_times[0])
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



class MyScheduler(BaseScheduler):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)

    def step(self, cases, current_active_agents=None):
        """
        Step through the agents, activating each agent in a dynamic subset.
        """
        self.do_each(method="step", agent_keys=current_active_agents, cases=cases)
        self.steps += 1
        self.time += 1

    def get_agent_count(self):
        """
        Returns the current number of active agents in the model.
        """
        return len(self._agents)
    
    def do_each(self, method, cases, agent_keys=None, shuffle=False):
        agent_keys_ = [agent_keys[0]] 
        if agent_keys_ is None:
            agent_keys_ = self.get_agent_keys()
        if shuffle:
            self.model.random.shuffle(agent_keys_)
        for agent_key in agent_keys_:
            if agent_key in self._agents:
                getattr(self._agents[agent_key], method)(self, agent_keys, cases)