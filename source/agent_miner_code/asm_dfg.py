import datetime
import networkx as nx
import pandas as pd
import pm4py
from pm4py.objects.log.util import dataframe_utils
from pm4py.algo.filtering.dfg import dfg_filtering
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.visualization.dfg import visualizer as dfg_visualization

#USED
def dfg_obj2nx(dfg_obj):
    # create Nx graph for the agent dfg dictionary
    dfg_nx = nx.DiGraph()
    for activity_type in dfg_obj['activity_types']:
        # any agent dfg contains not more than one node for any activity type in the input event log
        node_id = activity_type
        dfg_nx.add_node(node_id)
        dfg_nx.nodes[node_id]['label'] = activity_type
        count = dfg_obj['activity_types'][activity_type]
        dfg_nx.nodes[node_id]['count'] = count
        if activity_type in dfg_obj['input_activity_types']:
            in_port = 'in'
            dfg_nx.nodes[node_id]['in_port'] = True
        else:
            in_port = ''
            dfg_nx.nodes[node_id]['in_port'] = False
        if activity_type in dfg_obj['output_activity_types']:
            out_port = 'out'
            dfg_nx.nodes[node_id]['out_port'] = True
        else:
            out_port = ''
            dfg_nx.nodes[node_id]['out_port'] = False
        fill_color = "#FFA500" if in_port==out_port else "#CC99FF"
        dfg_nx.nodes[node_id]['graphics'] = {'w':150.0,'h':50.0, 'fill':fill_color}
        sub_label = f"{count} {in_port} {out_port}".strip()
        labels = [(activity_type,"t","center"),(sub_label,"b")]
        dfg_nx.nodes[node_id]['LabelGraphics'] = [{'text':l[0],'anchor':l[1]} for l in labels]
    for from_n,to_n in dfg_obj['control_flows']:
        dfg_nx.add_edge(from_n,to_n, label = dfg_obj['control_flows'][(from_n,to_n)])
    return dfg_nx

#USED
def discover_dfg_pm4py(dfg_id,log_pm4py,activity_frequency_filter=1):
    # log_pm4py.properties[constants.PARAMETER_CONSTANT_CASEID_KEY] = trace_id_field
    dfg_pm4py, start_activities, end_activities = pm4py.discover_directly_follows_graph(log_pm4py)
    activities_count = pm4py.get_event_attribute_values(log_pm4py, "concept:name")
    dfg_pm4py, start_activities, end_activities, activities_count = dfg_filtering.filter_dfg_on_activities_percentage(
        dfg_pm4py, 
        start_activities, 
        end_activities, 
        activities_count, 
        activity_frequency_filter
        )
    total_event_count = 0
    for activity_type in activities_count:
        total_event_count += activities_count[activity_type]
    dfg_obj = {
        'agent_types':{dfg_id},
        'event_count': total_event_count,
        'activity_count': len(activities_count),
        'flow_count': len(dfg_pm4py),
        'activity_types': activities_count,
        'control_flows': dfg_pm4py,
        'input_activity_types': start_activities,
        'output_activity_types': end_activities
    }
    return dfg_obj

def make_all_activities_in_out(dfg_obj):
    for activity_type in dfg_obj['activity_types']:
        dfg_obj['input_activity_types'][activity_type]=1
        dfg_obj['output_activity_types'][activity_type]=1
    return dfg_obj

def viz_dfg(dfg_obj,output_file_name_base,more_viz=False):
    print(datetime.datetime.now(),f"Writing DFG NX GML to file {output_file_name_base}.gml ...")
    dfg_nx = dfg_obj2nx(dfg_obj)
    nx.write_gml(dfg_nx,output_file_name_base+".gml")
    if more_viz:
        print(datetime.datetime.now(),f"Adding nodes with no edges to DFG ...")
        nodes_with_edges = set()
        for node_from,node_to in dfg_obj['control_flows']:
            nodes_with_edges.add(node_from)
            nodes_with_edges.add(node_to)
        nodes_with_no_edges = {node for node in dfg_obj['activity_types'] if not node in nodes_with_edges}
        for node in nodes_with_no_edges:
            dfg_obj['control_flows'][(node,node)] = 0
        print(datetime.datetime.now(),f"Writing DFG GVIZ to file {output_file_name_base}.pdf ...")
        gviz = dfg_visualization.apply(
            dfg_obj['control_flows'],
            activities_count=dfg_obj['activity_types'],
            parameters = {'start_activities':dfg_obj['input_activity_types'],'end_activities':dfg_obj['output_activity_types'],'format':"pdf"},
            variant=dfg_visualization.Variants.FREQUENCY
            )
        dfg_visualization.save(gviz,output_file_name_base+".pdf")
        for node in nodes_with_no_edges:
            dfg_obj['control_flows'].pop((node,node))

#USED
def dfg_nx2obj(agent_id,dfg_nx):
    dfg_obj = {
        'agent_types':{agent_id},
        'event_count': -1,
        'activity_count': -1,
        'flow_count': len(dfg_nx.edges),
        'input_activity_types':{},
        'output_activity_types':{},
        'activity_types':{},
        'control_flows':{}
    }
    for node in dfg_nx.nodes:
        labels = dfg_nx.nodes[node]['LabelGraphics']
        labels = [lg for lg in labels if 'text' in lg]
        # assumption: each node has 2 labels
        activity_type = str(labels[0]['text'])
        sub_label_parts = labels[1]['text'].split(' ')
        sub_label_parts = [l.strip() for l in sub_label_parts if len(l)>0]
        count = int(sub_label_parts[0])
        dfg_obj['activity_types'][activity_type] = count
        if len(sub_label_parts)>2 or (len(sub_label_parts)>1 and sub_label_parts[1]=='in'):
            dfg_obj['input_activity_types'][activity_type] = count
        if len(sub_label_parts)>2 or (len(sub_label_parts)>1 and sub_label_parts[1]=='out'):
            dfg_obj['output_activity_types'][activity_type] = count
    for from_id,to_id in dfg_nx.edges:
        dfg_obj['control_flows'][(from_id,to_id)] = int(dfg_nx.edges[from_id,to_id]['label'])
    return dfg_obj    