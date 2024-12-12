import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.organizational_mining.sna import algorithm as sna
from pm4py.algo.organizational_mining.sna import util

dir = os.getcwd()+"\\output\\"
mas_agent_traces_file_name = dir+"mas_agent_traces.csv"
agent_dfg_file_name_base = dir+"agent_inst_dfg_"

dfg_obj_template = {
        'agent_types':{}, # multiset of agent type labels
        'activity_types':{}, # multiset of pairs (agent type label,activity type label)
        'control_flows':{}, # multiset of pairs of pairs ((agent type label,activity type label),(agent type label,activity type label))
        'input_activity_types':set(), # map of pairs (agent type label,activity type label) to sets of pairs (agent type label,activity type label)
        'output_activity_types':set(), # map of pairs (agent type label,activity type label) to sets of pairs (agent type label,activity type label)
    }

#USED
_zero_dfg_nx = nx.DiGraph()
def get_dfg_normalized_graph_edit_distance_nx(dfg1_nx, dfg2_nx):
    def node_match_nx(node1_nx, node2_nx):
        return node1_nx['label'] == node2_nx['label']
    
    def edge_match_nx(edge1_nx,edge2_nx):
        return (edge1_nx['from_node_id'] == edge2_nx['from_node_id']) and (edge1_nx['to_node_id'] == edge2_nx['to_node_id'])
    for from_n,to_n in dfg1_nx.edges:
        dfg1_nx.edges[from_n,to_n]['from_node_id'] = from_n
        dfg1_nx.edges[from_n,to_n]['to_node_id'] = to_n
    for from_n,to_n in dfg2_nx.edges:
        dfg2_nx.edges[from_n,to_n]['from_node_id'] = from_n
        dfg2_nx.edges[from_n,to_n]['to_node_id'] = to_n
    deg01 = nx.algorithms.similarity.graph_edit_distance(_zero_dfg_nx,dfg1_nx,node_match=node_match_nx,edge_match=edge_match_nx,timeout=0.5)
    deg02 = nx.algorithms.similarity.graph_edit_distance(_zero_dfg_nx,dfg2_nx,node_match=node_match_nx,edge_match=edge_match_nx,timeout=0.5)
    deg12 = nx.algorithms.similarity.graph_edit_distance(dfg2_nx,dfg1_nx,node_match=node_match_nx,edge_match=edge_match_nx,timeout=0.5)
    if (deg12 is None) or (deg01 is None) or (deg02 is None):
        normalized_deg12 = 1
    else:    
        normalized_deg12 = deg12 / (deg01+deg02)
    return normalized_deg12

def get_dfg_normalized_graph_edit_distance_activity_types(dfg1_obj, dfg2_obj):
    #This is my idea for clustering. Let DFi be the set of activity types of agent i. 
    # Then, the distance between agents 1 and 2 is: 1-max((|DF1 intersect DF2|/|DF1|), (|DF1 intersect DF2|/|DF2|))
    act_intersect = [act for act in dfg1_obj['activity_types'] if act in dfg2_obj['activity_types']]
    act_intersect_count = len(act_intersect)
    act_count_dfg1 = len(dfg1_obj['activity_types'])
    act_count_dfg2 = len(dfg2_obj['activity_types'])
    if act_count_dfg1>0 and act_count_dfg2>0:
        dist = 1 - max(act_intersect_count/act_count_dfg1,act_intersect_count/act_count_dfg2)
    else:
        dist = 0
    return (dist,act_intersect_count)

def get_dfg_normalized_graph_edit_distance_artem(dfg1_obj, dfg2_obj):
    #This is Artem's idea for clustering. Let DFi be the set of DF pairs of agent i. 
    # Then, the distance between agents 1 and 2 is: 1-max((|DF1 intersect DF2|/|DF1|), (|DF1 intersect DF2|/|DF2|))

    e_intersect = [e for e in dfg1_obj['control_flows'] if e in dfg2_obj['control_flows']]
    e_intersect.extend([('_i_',i) for i in dfg1_obj['input_activity_types'] if i in dfg2_obj['input_activity_types']])
    e_intersect.extend([(o,'_o_') for o in dfg1_obj['output_activity_types'] if o in dfg2_obj['output_activity_types']])
    e_count_intersect = len(e_intersect)
    e_count_dfg1 = len(dfg1_obj['control_flows']) + len(dfg1_obj['input_activity_types']) + len(dfg1_obj['output_activity_types'])
    e_count_dfg2 = len(dfg2_obj['control_flows']) + len(dfg2_obj['input_activity_types']) + len(dfg2_obj['output_activity_types'])
    if e_count_dfg1>0 and e_count_dfg2>0:
        dist = 1 - max(e_count_intersect/e_count_dfg1,e_count_intersect/e_count_dfg2)
    else:
        dist = 0
    return (dist,e_count_intersect)

#_NOT-USED
def get_dfg_distance_na_kalenke(dfg1_obj, dfg2_obj):
    # assuming agent_id===agent_type
    #find distinct nodes
    i = 0
    sim = 0
    for activity_type1 in dfg1_obj['activity_types']:
        agent1,activity1 = activity_type1
        match = False
        for activity_type2 in dfg2_obj['activity_types']:
            agent2,activity2 = activity_type2
            if (activity1 == activity2):
                match = True
                sim = sim + 1
        if (match == False):
            i = i + 1
    for activity_type2 in dfg2_obj['activity_types']:
        agent2,activity2 = activity_type2
        match = False
        for activity_type1 in dfg1_obj['activity_types']:
            agent1,activity1 = activity_type1
            if (activity2 == activity1):
                match = True
        if (match == False):
            i = i + 1
    #find distinct edges
    j = 0
    for flow1 in dfg1_obj['control_flows']:
        (from_agent1,from_activity1),(to_agent1,to_activity1) = flow1
        match = False
        for flow2 in dfg2_obj['control_flows']:
            (from_agent2,from_activity2),(to_agent2,to_activity2) = flow2
            if ((from_activity1 == from_activity2) & (to_activity1 == to_activity2)):
                match = True
                sim = sim + 1
        if (match == False):
            j = j + 1
    for flow2 in dfg2_obj['control_flows']:
        (from_agent2,from_activity2),(to_agent2,to_activity2) = flow2
        match = False
        for flow1 in dfg1_obj['control_flows']:
            (from_agent1,from_activity1),(to_agent1,to_activity1) = flow1
            if ((from_activity1 == from_activity2) & (to_activity1 == to_activity2)):
                match = True
        if (match == False):
            j = j + 1
    dist = (i + j)/(i + j + sim)
    return dist

#USED
def discover_agent_dfg_distance_graph(agent_dfg_map):
    # calculate DFG distance among agent instances in form of a map (agent1,agent2) -> distance_betwen_agent1_agent2
    dfg_dist_map = {}
    dfg_intersect_map = {}
    agent_with_dedicated_activities_list = []
    for agent1_id in agent_dfg_map:
        dfg1_total_intersect=0
        for agent2_id in agent_dfg_map:
            ged = None
            intersect_count = None
            if (agent2_id,agent1_id) in dfg_dist_map:
                ged = dfg_dist_map[(agent2_id,agent1_id)]
                intersect_count = dfg_intersect_map[(agent2_id,agent1_id)]
            elif agent1_id==agent2_id:
                ged = 0
                intersect_count = -1
            else:
                dfg1_obj = agent_dfg_map[agent1_id]
                dfg2_obj = agent_dfg_map[agent2_id]
                # print(f"Calculating distance between {agent1_id} and {agent2_id} ...")
                # ged = get_dfg_normalized_graph_edit_distance_nx(dfg1_nx,dfg2_nx)
                ged,intersect_count = get_dfg_normalized_graph_edit_distance_artem(dfg1_obj,dfg2_obj)
                #ged,intersect_count = get_dfg_normalized_graph_edit_distance_activity_types(dfg1_obj,dfg2_obj)
            dfg_dist_map[(agent1_id,agent2_id)] = ged
            dfg_intersect_map[(agent1_id,agent2_id)] = intersect_count
            dfg1_total_intersect=dfg1_total_intersect+intersect_count
        if dfg1_total_intersect<0:
            print("___ Agent with dedicated activities: ", agent1_id)    
            agent_with_dedicated_activities_list.append(agent1_id)
    return (dfg_dist_map,dfg_intersect_map,agent_with_dedicated_activities_list)

#USED
def save_agent_dfg_distance_matrix(dfg_distance_file_name_base,dfg_dist_map,dfg_intersect_map):
    pairs_file_name = dfg_distance_file_name_base+"_pairs.csv"
    agent_matrix = {}
    a_degree_map = {}
    with open(pairs_file_name,"w") as pairs_file:
        print("agent1_id,agent2_id,distance",file=pairs_file)
        for a1_id,a2_id in dfg_dist_map:
            dist = dfg_dist_map[(a1_id,a2_id)]
            intersect_count = dfg_intersect_map[(a1_id,a2_id)]
            print(f"{str(a1_id)},{str(a2_id)},{str(dist)},{str(intersect_count)}",file=pairs_file)
            if not a1_id in agent_matrix:
                agent_matrix[a1_id] = {}
            agent_matrix[a1_id][a2_id] = (dist,intersect_count)
            if not a1_id in a_degree_map: a_degree_map[a1_id] = 0
            a_degree_map[a1_id] = a_degree_map[a1_id] + dist
    a_degree_list = [{'agent_id':a,'degree_of_separation':a_degree_map[a]} for a in a_degree_map]
    a_degree_df = pd.DataFrame(a_degree_list)
    a_degree_df = a_degree_df.sort_values(by='degree_of_separation',ascending=False)
    a_degree_df.to_csv(dfg_distance_file_name_base+"dos.csv")
    matrix_file_name = dfg_distance_file_name_base+"_matrix.csv"
    matrix_header = ""
    matrix_lines = {}
    for a_id in agent_matrix:
        matrix_header += f",{str(a_id)}"
        matrix_lines[a_id] = str(a_id)
        for a_id_col in agent_matrix:
            dist,intersect_count = agent_matrix[a_id][a_id_col]
            matrix_lines[a_id] += f",({dist};{intersect_count})"
    with open(matrix_file_name,"w") as matrix_file:
        # write the file hader
        print(matrix_header,file=matrix_file)
        for a_id in matrix_lines:
            print(matrix_lines[a_id],file=matrix_file)

def read_agent_dfg_distance_matrix(dfg_distance_file_name_base):
    pairs_file_name = dfg_distance_file_name_base+"_pairs.csv"
    map_df = pd.read_csv(pairs_file_name,sep=',')
    dfg_dist_map = {}
    for idx,row in map_df.iterrows():
        dfg_dist_map[(row['agent1_id'],row['agent2_id'])] = row['distance']
    return dfg_dist_map

#USED
def save_cluster_instance_map_csv(inst_cluster_map,output_file_name):
    cluster_instance_map = {}
    for inst_id in inst_cluster_map:
        cl_id = inst_cluster_map[inst_id]
        if cl_id not in cluster_instance_map:
            cluster_instance_map[cl_id] = []
        cluster_instance_map[cl_id].append(inst_id)
    with open(output_file_name,"w") as log_file:
        # write the log ehader
        print("cluster_id,instance_id",file=log_file)
        for cluster_id in cluster_instance_map:
           print_line = f"{str(cluster_id)},"
           for inst_id in cluster_instance_map[cluster_id]:
               print_line += f"{str(inst_id)};"
           print(print_line,file=log_file)    
    return cluster_instance_map

#USED
def read_cluster_instance_map_csv(map_csv_file_name):
    map_df = pd.read_csv(map_csv_file_name,sep=',')
    cluster_inst_map = {}
    for idx,row in map_df.iterrows():
        cluster_inst_map[row['cluster_id']] = row['instance_id']
    return cluster_inst_map

#USED
def read_instance_cluster_map_csv(map_csv_file_name):
    map_df = pd.read_csv(map_csv_file_name,sep=',')
    inst_cluster_map = {}
    for idx,row in map_df.iterrows():
        inst_cluster_map[row['agent_inst']] = row['agent_cluster']
    return inst_cluster_map

#USED
def group_agents_to_clusters(dfg_dist_map,max_dist=0.99):
    # print(f"dfg_dist_map: {dfg_dist_map}")
        # build agent dfg distance graph (nodes are agent instances, edge weights are distances between agent DFGs)
    addg_nx = nx.DiGraph()
    for a1,a2 in dfg_dist_map:
        # print(f"a1: {a1}, a2: {a2}")
        if not addg_nx.has_node(a1): addg_nx.add_node(a1)
        if not addg_nx.has_node(a2): addg_nx.add_node(a2)
        if (not a1==a2) and (not dfg_dist_map[(a1,a2)] > max_dist):
            if not addg_nx.has_edge(a1,a2):
                addg_nx.add_edge(a1,a2,weight=dfg_dist_map[(a1,a2)])
                print(f"added edge from {a1} to {a2} with weight {dfg_dist_map[(a1,a2)]}")
    print(f"Number of edges in graph: {addg_nx.number_of_edges()}")
    # for u, v, weight in addg_nx.edges(data='weight'):
    #     print(f"Edge ({u}, {v}) has weight: {weight}")
    if len(addg_nx.edges)>0:
        agent_clusters_list = nx.algorithms.community.greedy_modularity_communities(addg_nx,weight='weight')
    else:
        agent_clusters_list = [{a} for a in addg_nx.nodes]
    inst_cluster_map = {}
    cl_idx = 1
    for cl_inst_set in agent_clusters_list:
        if len(cl_inst_set)>1:
            cluster_id = 'a'+str(cl_idx)
            cl_idx += 1
        else:
            cluster_id = list(cl_inst_set)[0]
        for inst_id in cl_inst_set:
            inst_cluster_map[inst_id] = cluster_id
    return inst_cluster_map

def viz_agent_dfg_graph(addg_nx,output_file_name_base):
    g_nx = nx.DiGraph()

    a_abbr_map = {}
    a_idx = 0
    with open(output_file_name_base+"_abbr.csv","w") as log_file:
        # write the log ehader
        print("inst_abbr,inst_full_name",file=log_file)
        for a_full_name in addg_nx.nodes:
            a_idx = a_idx + 1
            a_abbr = "a"+str(a_idx)
            a_abbr_map[a_full_name] = a_abbr
            print_line = f"{a_abbr},{a_full_name}"
            print(print_line,file=log_file)    

    for a1,a2 in addg_nx.edges:
        dist = round(addg_nx.edges[a1,a2]['weight'],2)
        w = (1 - dist)/10
        g_nx.add_edge(a_abbr_map[a1],a_abbr_map[a2],label=dist,weight=w)
    pos=nx.spring_layout(g_nx,seed = 7)
    nx.draw(g_nx,pos,node_size=1500)
    # node labels
    nx.draw_networkx_labels(g_nx, pos, font_family="sans-serif", font_size=9, font_color='w')
    # edge weight labels
    edge_labels = nx.get_edge_attributes(g_nx,"label")
    nx.draw_networkx_edge_labels(g_nx, pos, edge_labels)
    ax = plt.gca()
    #ax.margins(0.08)
    plt.axis("off")
    #plt.tight_layout()
    plt.savefig(output_file_name_base+".svg")