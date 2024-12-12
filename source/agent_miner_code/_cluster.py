import pandas as pd
import networkx as nx
import pm4py
import os
from source.agent_miner_code import asm_data as da
import datetime as dt
from source.agent_miner_code import asm_dfg as dfg
from source.agent_miner_code import asm_cluster as cl

def discover_agent_dfg_dist(log_dir="bpic_",filter_ver="aol1"):
    input_dir = os.path.join(os.getcwd(),log_dir,filter_ver,"_no_clustering_")
    agent_dfgs_dir = os.path.join(input_dir,"agents")
    dfg_summary_df = pd.read_csv(os.path.join(agent_dfgs_dir,'agents_dfg_summary.csv'),sep=",")
    agent_dfg_map = {}
    for idx,row in dfg_summary_df.iterrows():
        agnet_inst_id = row['agent_inst']
        a_dfg_nx = nx.read_gml(os.path.join(agent_dfgs_dir,f'{agnet_inst_id}_dfg.gml'))
        a_dfg_obj = dfg.dfg_nx2obj(agnet_inst_id,a_dfg_nx)
        agent_dfg_map[agnet_inst_id] = a_dfg_obj
    dfg_dist_map,dfg_intersect_map,agent_with_dedicated_activities_list = cl.discover_agent_dfg_distance_graph(agent_dfg_map)
    return dfg_dist_map,dfg_intersect_map,agent_with_dedicated_activities_list
    
def _cluster_agent_instances(log_dir,filter_ver,cluster_ver,max_dfg_dist=1.0,inst_cluster_map=None):
    print(dt.datetime.now(),f"Agent clustering finished for {log_dir} : {filter_ver} : {cluster_ver}!")
    filter_dir = os.path.join(os.getcwd(),log_dir,filter_ver)
    inst_log_file_path = os.path.join(filter_dir,"_no_clustering_","_log_aol.csv")
    inst_mas_log_df = pd.read_csv(inst_log_file_path,sep=",")
    print(dt.datetime.now(),f"Agent instance log '{inst_log_file_path}' loaded!")

    # print(inst_mas_log_df)
   
    cluster_dir = os.path.join(os.getcwd(),log_dir,filter_ver,cluster_ver)
    os.system(f"mkdir {cluster_dir}")
    inst_cluster_map_updated = None
    # print(f" received cluster map: {inst_cluster_map}")
    cluster_set = set() if inst_cluster_map is None else {inst_cluster_map[inst_id] for inst_id in inst_cluster_map}
    while inst_cluster_map is None or (max_dfg_dist>0 and len(inst_cluster_map)>1 and len(cluster_set)<2):
        print(f"cluster map: {inst_cluster_map}")
        # do automated clustering
        # there should be at least 2 clusters
        dfg_dist_map,dfg_intersect_map,agent_with_dedicated_activities_list = discover_agent_dfg_dist(log_dir=log_dir,filter_ver=filter_ver)
        cl.save_agent_dfg_distance_matrix(os.path.join(cluster_dir,"addg"),dfg_dist_map,dfg_intersect_map)
        inst_cluster_map = cl.group_agents_to_clusters(dfg_dist_map,max_dist=max_dfg_dist)
        cluster_set = {inst_cluster_map[inst_id] for inst_id in inst_cluster_map}
        max_dfg_dist = max_dfg_dist -0.01 
    agent_gr = inst_mas_log_df.groupby(by=["agent_id"],sort=False)
    inst_cluster_map_updated = {}
    for agent_id,a_df in agent_gr:
        # added by Lukas
        agent_id = agent_id[0]

        if agent_id in inst_cluster_map:    
            inst_cluster_map_updated[agent_id] = inst_cluster_map[agent_id]
        else:
            inst_cluster_map_updated[agent_id]="_others_"

    cluster_inst_map = cl.save_cluster_instance_map_csv(inst_cluster_map_updated,os.path.join(cluster_dir,"cluster_inst_map.csv"))
    print(dt.datetime.now(),f"Adding agent clusters to events ...")
    def make_agent_evt(evt):
        inst_id = evt['agent_id']
        cluster_id = inst_cluster_map_updated[inst_id]
        evt['agent_inst_id'] = inst_id
        evt['agent_id'] = cluster_id
        evt['agent_activity_type'] = f"{cluster_id}|{evt['activity_type']}"
        return evt
    cluster_mas_log_df = inst_mas_log_df.apply(make_agent_evt,axis=1)    

    clustered_log_name_base = os.path.join(cluster_dir,"_log_aol")
    print(f"Writing clustered event selection with activity-only labels (aol) {clustered_log_name_base} ...")
    log_df = pm4py.format_dataframe(cluster_mas_log_df, case_id='case_id', activity_key='activity_type', timestamp_key='timestamp')
    # added by Lukas
    log_df['timestamp'] = pd.to_datetime(log_df['timestamp'], format='mixed')
    
    log_pm4py = pm4py.convert_to_event_log(log_df)
    log_pm4py = da.add_fragments_to_events(log_pm4py)
    log_df = pm4py.convert_to_dataframe(log_pm4py)
    da.save_xes_log(log_df,clustered_log_name_base)
    clustered_log_name_base = os.path.join(cluster_dir,"_log_aal")
    print(f"Writing clustered event selection with agent-activity labels (aal) {clustered_log_name_base} ...")
    log_df = pm4py.format_dataframe(log_df, case_id='case_id', activity_key='agent_activity_type', timestamp_key='timestamp')
    da.save_xes_log(log_df,clustered_log_name_base)

    print(dt.datetime.now(),f"Agent clustering finished for {log_dir} : {filter_ver} : {cluster_ver}!")
    return cluster_mas_log_df
