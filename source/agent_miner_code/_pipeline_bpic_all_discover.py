import _pipeline as p
import _filter as f
import asm_config as config
import pm4py
import os
import pandas as pd

frequency_coverage = 100

data_dir = "datasets/Credit_Application"
file_name = 'Credit_Application.csv'
col_map={
    'timestamp':'start_timestamp',
    'case_id':'case_id',
    'activity_type':'Activity',
    'agent_id':'Resource'
    }

data_dir = "datasets/Loan_Application"
file_name = 'LoanApp.csv'
col_map={
    'timestamp':'start_time',
    'case_id':'case_id',
    'activity_type':'activity',
    'agent_id':'resource'
    }
data_dir = "datasets/bpic_2012_w"
file_name = 'BPIC_2012_W_train.csv'
col_map={
    'timestamp':'time:timestamp',
    'case_id':'case:concept:name',
    'activity_type':'Activity',
    'agent_id':'Resource'
    }
bpic_log_dir_list = [(data_dir,frequency_coverage)]
log_percent_list = [frequency_coverage]

run_name="loan_app_roles"
# am_threshold_list = [(round(0.1+i*0.1,1),round(0.9-i*0.1,1)) for i in range(0,10)]
am_threshold_list = [(1.0,0.0)] # frequency filter is set to 1 and noise threshold is set to 0 -> use all available cases
# soa_threshold_list = [round(0.9-i*0.1,1) for i in range(0,10)]

config._in_alg = pm4py.algo.discovery.inductive.algorithm.Variants.IMf
config._a_alg = "DFG-WN"
config._soa_alg = pm4py.algo.discovery.inductive.algorithm.Variants.IMf

def preprocess_evt(evt):
        r = evt[col_map['agent_id']]
        if str(r)=='nan' or str(r)=='NaN' or r is None:
            evt[col_map['agent_id']] = "r"  + '_' + evt[col_map['activity_type']]
        else:
            evt[col_map['agent_id']] = "r"+str(r)
        return evt
df0 = f._preprocess_inst_log(
    log_dir=data_dir,
    orig_log_file_name=file_name,
    separator=',',
    col_map=col_map,
    preproces_evt_callback=preprocess_evt
    )
# added from Lukas
p._run_filter(example_dir=data_dir,run_log_percent_list=log_percent_list)

for bpic_log_dir,variant_frequency_cover_percent in bpic_log_dir_list:
    dev_params_dict_am = {
        'run_name':run_name,
        'example_dir':bpic_log_dir,
        'manual_inst_cluster_map_map':{},
        'run_threshold_list':am_threshold_list,
        'run_log_percent_list':[variant_frequency_cover_percent],
        'run_cluster_max_dist_list':[1.0], # was [1.0] before
        'label_type':'aol',
        'more_viz':False
    }

    # define cluster map manually resource-based
    # filter_v = "aol"+str(variant_frequency_cover_percent)
    # filter_dir = os.path.join(os.getcwd(),bpic_log_dir,filter_v)
    # inst_log_file_path = os.path.join(filter_dir,"_no_clustering_","_log_aol.csv")
    # inst_mas_log_df = pd.read_csv(inst_log_file_path,sep=",")
    # agents = inst_mas_log_df['agent_id'].unique()
    # num_agents = len(agents)
    # # Dynamically generate cluster identifiers based on the number of agents
    # clusters = [f'a{i+1}' for i in range(num_agents)]
    # # Create the mapping from agents to clusters
    # manual_inst_cluster_map_map = {
    # ''.join(clusters): {agent: clusters[i % num_agents] for i, agent in enumerate(agents)}
    # }
    # dev_params_dict_am['manual_inst_cluster_map_map'] = manual_inst_cluster_map_map

    # define cluster map manually role-based
    # mapping = {'r_Credit application received': 'a1', 'rClerk-000001': 'a2', 'rClerk-000002': 'a2', 'rCredit Officer-000001': 'a3', 'rCredit Officer-000002': 'a3', 'rClerk-000003': 'a2', 'rCredit Officer-000003': 'a3', 'rSystem-000001': 'a4', 'r_Credit application processed': 'a5'}
    # manual_inst_cluster_map_map={'a1a2a3a4a5':mapping}
    # dev_params_dict_am['manual_inst_cluster_map_map'] = manual_inst_cluster_map_map

    # Discover AM
    # dev_params_dict_am['label_type'] = 'aol'
    print(f"params: {dev_params_dict_am}")
    p._run_cluster(example_dir=data_dir, 
                   run_log_percent_list=dev_params_dict_am['run_log_percent_list'], 
                   manual_inst_cluster_map_map=dev_params_dict_am['manual_inst_cluster_map_map'], 
                   auto_cluster_max_dist_list=dev_params_dict_am['run_cluster_max_dist_list']
                   )
    print("FINISHED CLUSTERING")
    # # p._run_discover_am(**dev_params_dict_am)
    # # p._run_completed(bpic_log_dir+run_name+"_discovery_am_aol")
    # # print("FINISHED DISCOVERY")
    # # p._run_viz_am(run_name=run_name,example_dir=data_dir,
    # #             manual_inst_cluster_map_map=dev_params_dict_am['manual_inst_cluster_map_map'],
    # #             run_log_percent_list=dev_params_dict_am['run_log_percent_list'],
    # #             run_cluster_max_dist_list=dev_params_dict_am['run_cluster_max_dist_list'],
    # #             run_threshold_list=dev_params_dict_am['run_threshold_list'],
    # #             label_type='aol',
    # #             more_viz=False,
    # #             )
    # # print("FINISHED VISUALIZATION")

    # # ### not necessary for discovery
    # # p._run_eval_am_pm4py(run_name=run_name,example_dir=data_dir,
    # #                      manual_inst_cluster_map_map=dev_params_dict_am['manual_inst_cluster_map_map'],
    # #                      run_log_percent_list=dev_params_dict_am['run_log_percent_list'],
    # #                      run_cluster_max_dist_list=dev_params_dict_am['run_cluster_max_dist_list'],
    # #                      run_threshold_list=dev_params_dict_am['run_threshold_list'],
    # #                      label_type='aol',
    # #                      more_viz=False)
    # # ###

    # dev_params_dict_am['label_type'] = 'aal'
    # p._run_discover_am(**dev_params_dict_am)
    # p._run_viz_am(run_name=run_name,example_dir=data_dir,
    #             manual_inst_cluster_map_map=dev_params_dict_am['manual_inst_cluster_map_map'],
    #             run_log_percent_list=dev_params_dict_am['run_log_percent_list'],
    #             run_cluster_max_dist_list=dev_params_dict_am['run_cluster_max_dist_list'],
    #             run_threshold_list=dev_params_dict_am['run_threshold_list'],
    #             label_type='aal',
    #             more_viz=False,
    #             )