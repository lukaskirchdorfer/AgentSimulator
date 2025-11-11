import os
from source.train_test_split import split_data
from source.discovery import discover_simulation_parameters
from source.simulation import simulate_process
class AgentSimulator:
    def __init__(self, params):
        self.params = params

    def execute_pipeline(self):
        self.df_train, self.df_test, self.num_cases_to_simulate, self.df_val, self.num_cases_to_simulate_val = self._split_log()
        
        # discover basic simulation parameters
        self.df_train, self.simulation_parameters = discover_simulation_parameters(
            self.df_train, 
            self.df_test, 
            self.df_val, 
            self.data_dir, 
            self.num_cases_to_simulate, 
            self.num_cases_to_simulate_val,
            self.params['determine_automatically'],
            self.params['central_orchestration'],
            self.params['discover_extr_delays']
        )

        print(f"agent to resource: {self.simulation_parameters['agent_to_resource']}")

        # simulate process
        simulate_process(self.df_train, self.simulation_parameters, self.data_dir, self.params['num_simulations'])

    def _split_log(self):
        """
        Split the log into training, testing and validation data.
        """
        def get_validation_data(df):
            df_sorted = df.sort_values(by=['case_id', 'start_timestamp'])
            total_cases = df_sorted['case_id'].nunique()
            twenty_percent = int(total_cases * 0.2)
            last_20_percent_case_ids = df_sorted['case_id'].unique()[-twenty_percent:]
            df_val = df_sorted[df_sorted['case_id'].isin(last_20_percent_case_ids)]
            
            return df_val
        
        file_name = os.path.splitext(os.path.basename(self.params['PATH_LOG']))[0]
        if self.params['determine_automatically']:
            print("Choice for architecture and extraneous delays will be determined automatically")
            file_name_extension = 'main_results'
        else:
            if self.params['central_orchestration']:
                file_name_extension = 'orchestrated'
            else:
                file_name_extension = 'autonomous'
        if self.params['train_and_test']:
            df_train, df_test, num_cases_to_simulate = split_data(self.params['PATH_LOG'], self.params['column_names'], self.params['PATH_LOG_test'])
        else:
            df_train, df_test, num_cases_to_simulate = split_data(self.params['PATH_LOG'], self.params['column_names'])

        self.data_dir = os.path.join(os.getcwd(), "simulated_data", "ATKDE", file_name, file_name_extension)

        df_val = get_validation_data(df_train)
        num_cases_to_simulate_val = len(set(df_val['case_id']))

        return df_train, df_test, num_cases_to_simulate, df_val, num_cases_to_simulate_val
