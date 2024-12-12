from source.agent_miner_code import _pipeline as p
from source.agent_miner_code import _filter as f
from source.agent_miner_code import asm_config as config
import pm4py
import os
import pandas as pd

def run_agent_miner(data_dir, file_name, roles):
    col_map = {
        'timestamp':'start_timestamp',
        'case_id':'case_id',
        'activity_type':'activity_name',
        'agent_id':'resource'
    }
    run_name="roles"
    am_threshold_list = [(1.0,0.0)] # frequency filter is set to 1 and noise threshold is set to 0 -> use all available cases
    config._in_alg = pm4py.algo.discovery.inductive.algorithm.Variants.IMf
    config._a_alg = "DFG-WN"
    config._soa_alg = pm4py.algo.discovery.inductive.algorithm.Variants.IMf
    frequency_coverage = 100
    # bpic_log_dir_list = [(data_dir,frequency_coverage)]
    log_percent_list = [frequency_coverage]

    def preprocess_evt(evt):
        r = evt[col_map['agent_id']]
        if str(r)=='nan' or str(r)=='NaN' or r is None:
            evt[col_map['agent_id']] = "r"  + '_' + evt[col_map['activity_type']]
        else:
            evt[col_map['agent_id']] = "r"+str(r)
        return evt
    df0, agent_to_resource_map = f._preprocess_inst_log(
        log_dir=data_dir,
        orig_log_file_name=file_name,
        separator=',',
        col_map=col_map,
        preproces_evt_callback=preprocess_evt
        )
    p._run_filter(example_dir=data_dir,run_log_percent_list=log_percent_list)

    dev_params_dict_am = {
        'run_name':run_name,
        'example_dir':data_dir,
        'manual_inst_cluster_map_map':{},
        'run_threshold_list':am_threshold_list,
        'run_log_percent_list':[frequency_coverage],
        'run_cluster_max_dist_list':[], # was [1.0] before
        'label_type':'aol',
        'more_viz':False
    }

    # define cluster map manually role-based
    mapping = {}
    roles_short = ''
    for role, data in roles.items():
        role_short = 'a' + role.split(' ')[1]  # Convert "Role 1" to "r1", "Role 2" to "r2", etc.
        roles_short += role_short
        for agent in data['agents']:
            resource = agent_to_resource_map[agent]
            mapping[resource] = role_short
    print(mapping)
    # mapping = {'r_Credit application received': 'a1', 'rClerk-000001': 'a2', 'rClerk-000002': 'a2', 'rCredit Officer-000001': 'a3', 'rCredit Officer-000002': 'a3', 'rClerk-000003': 'a2', 'rCredit Officer-000003': 'a3', 'rSystem-000001': 'a4', 'r_Credit application processed': 'a5'}
    manual_inst_cluster_map_map={roles_short:mapping}
    dev_params_dict_am['manual_inst_cluster_map_map'] = manual_inst_cluster_map_map

    # discover AM
    p._run_cluster(example_dir=data_dir, 
                   run_log_percent_list=dev_params_dict_am['run_log_percent_list'], 
                   manual_inst_cluster_map_map=dev_params_dict_am['manual_inst_cluster_map_map'], 
                   auto_cluster_max_dist_list=dev_params_dict_am['run_cluster_max_dist_list']
                   )
    print("FINISHED CLUSTERING")

    # dev_params_dict_am['label_type'] = 'aal'
    p._run_discover_am(**dev_params_dict_am)
    p._run_viz_am(run_name=run_name,example_dir=data_dir,
                manual_inst_cluster_map_map=dev_params_dict_am['manual_inst_cluster_map_map'],
                run_log_percent_list=dev_params_dict_am['run_log_percent_list'],
                run_cluster_max_dist_list=dev_params_dict_am['run_cluster_max_dist_list'],
                run_threshold_list=dev_params_dict_am['run_threshold_list'],
                label_type=dev_params_dict_am['label_type'],
                more_viz=False,
                )

    # print(f"config out dir: {config.out_dir()}")

    path_to_agentminer_nets = config.out_dir()

    return path_to_agentminer_nets

