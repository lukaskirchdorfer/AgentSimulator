import pandas as pd
import datetime as dt
import pm4py

#USED
def save_xes_log(xes_log_df,log_file_name_base,save_xes = True):
    # Save CSV file
    xes_log_df.to_csv(log_file_name_base+'.csv',sep=',',float_format='%.2f',index=False)
    if save_xes:
        # Save XES XML file
        pm4py.write_xes(xes_log_df,log_file_name_base+".xes")
    return xes_log_df

#USED
def select_columns_xes_log(log_df,column_map):
    # rename columns to XES names
    columns_to_rename = {}
    for target_col in column_map:
        columns_to_rename[column_map[target_col]] = target_col
    log_df = log_df.rename(columns=columns_to_rename)
    # remove not selected columns from the log
    columns_to_drop = [colName for colName in log_df.columns if not colName in column_map]
    log_df = log_df.drop(columns=columns_to_drop)
    return log_df

def add_fragments_to_events(log_pm4py):
    events_with_fragments_list = []
    for trace in log_pm4py:
        case_id = trace.attributes['concept:name']
        prev_agent_id = None
        fragment_idx = 0
        for evt in trace:
            agent_id = evt['agent_id']
            if (not agent_id==prev_agent_id):
                fragment_idx += 1
                prev_agent_id = agent_id
            evt['fragment_id'] = f"{case_id}_{agent_id}_{fragment_idx}"
    return log_pm4py

def add_message_events_to_log_with_fragments_df(agent_trace_set_pm4py):
    evt_params = {'timestamp','case_id','activity_type','agent_id','fragment_id'}
    def add_msg_events(out_evt,in_evt,log_with_message_events_list):
        if out_evt is None and in_evt is None:
            return log_with_message_events_list
        if out_evt is None: 
            out_evt = {param:in_evt[param] for param in evt_params}
            out_evt['agent_id'] = '_env'
            out_evt['activity_type'] = '_start'
            out_evt['fragment_id'] = out_evt['case_id']
            out_evt['timestamp'] = in_evt['timestamp'] - dt.timedelta(seconds=3)
        if in_evt is None: 
            in_evt = {param:out_evt[param] for param in evt_params}
            in_evt['agent_id'] = '_env'
            in_evt['activity_type'] = '_finish'
            in_evt['fragment_id'] = in_evt['case_id']
            in_evt['timestamp'] = out_evt['timestamp'] + dt.timedelta(seconds=3)
        out_msg_evt = {param:out_evt[param] for param in evt_params}
        out_msg_evt['activity_type'] = f"_msg_out|{out_evt['agent_id']}|{out_evt['activity_type']}|{in_evt['agent_id']}|{in_evt['activity_type']}"
        out_msg_evt['timestamp'] = out_evt['timestamp'] + dt.timedelta(seconds=1)
        log_with_message_events_list.append(out_msg_evt)
        in_msg_evt = {param:in_evt[param] for param in evt_params}
        in_msg_evt['activity_type'] = f"_msg_in|{out_evt['agent_id']}|{out_evt['activity_type']}|{in_evt['agent_id']}|{in_evt['activity_type']}"
        in_msg_evt['timestamp'] = in_evt['timestamp'] - dt.timedelta(seconds=1)
        log_with_message_events_list.append(in_msg_evt)
        return log_with_message_events_list
    # we assume that log_pm4py fragment ids added to it
    log_with_message_events_list = []
    for a_trace in agent_trace_set_pm4py:
        prev_evt = None
        for orig_evt in a_trace:
            evt = {param:orig_evt[param] for param in evt_params}
            log_with_message_events_list.append(evt)
            if (prev_evt is None) or (not prev_evt['fragment_id']==evt['fragment_id']):
                add_msg_events(prev_evt,evt,log_with_message_events_list)
            prev_evt = evt
        evt = None
        add_msg_events(prev_evt,evt,log_with_message_events_list)        
    log_with_message_events_df = pd.DataFrame(log_with_message_events_list)
    return log_with_message_events_df

def add_clusters_to_events(inst_mas_log_df,inst_cluster_map):
    cluster_mas_log_df = inst_mas_log_df.copy()
    def set_cluster_id(evt):
        agent_id = evt['agent_id']
        evt['agent_inst_id'] = agent_id
        evt['agent_id'] = inst_cluster_map[agent_id]
        return evt
    cluster_mas_log_df = cluster_mas_log_df.apply(set_cluster_id,axis=1)    
    return cluster_mas_log_df

def create_interaction_log(mas_log_df,agent_trace_col_name='fragment_id'):
    agent_traces_over_mas_df = pm4py.format_dataframe(mas_log_df, case_id=agent_trace_col_name, activity_key='agent_activity_type', timestamp_key='timestamp')
    agent_traces_over_mas_pm4py = pm4py.convert_to_event_log(agent_traces_over_mas_df)
    hn_events_list = [f_trace[0] for f_trace in agent_traces_over_mas_pm4py]
    in_df = pd.DataFrame(hn_events_list)
    in_df = pm4py.format_dataframe(in_df, case_id='case_id', activity_key='agent_id', timestamp_key='timestamp')
    return pm4py.convert_to_event_log(in_df), in_df

def create_io_log(agent_trace_set_pm4py,agent_trace_col_name='fragment_id'):
    evt_params = {'timestamp','case_id','activity_type','agent_id','fragment_id'}
    io_events_list = []
    for a_trace in agent_trace_set_pm4py:
        i_evt = a_trace[0]
        o_evt = a_trace[-1]
        agent_id = i_evt['agent_id']
        i_act = i_evt['activity_type']
        o_act = o_evt['activity_type']
        io_evt = {param:o_evt[param] for param in evt_params}
        io_evt = {
            'agent_id':agent_id,
            'fragment_id':i_evt['fragment_id'],
            'case_id':i_evt['case_id'],
            'start_time':i_evt['timestamp'],
            'finish_time':o_evt['timestamp'],
            'activity_type':f"{agent_id}|{i_act}|{o_act}"
        }
        io_events_list.append(io_evt)
    io_log_df = pd.DataFrame(io_events_list)
    io_log_df = pm4py.format_dataframe(io_log_df, case_id='case_id', activity_key='activity_type', timestamp_key='start_time')
    return pm4py.convert_to_event_log(io_log_df)

def create_interaction_log_with_messages(log_with_msg_pm4py,agent_trace_col_name='fragment_id'):
    agent_traces_over_mas_df = pm4py.format_dataframe(pm4py.convert_to_dataframe(log_with_msg_pm4py), case_id=agent_trace_col_name, activity_key='activity_type', timestamp_key='timestamp')
    agent_traces_over_mas_pm4py = pm4py.convert_to_event_log(agent_traces_over_mas_df)
    evt_params = {'timestamp','case_id','activity_type','agent_id','fragment_id'}
    hn_events_list = []
    for a_trace in agent_traces_over_mas_pm4py:
        agent_evt = None
        for evt in a_trace:
            act =  evt['activity_type']
            act_parts = act.split("|")
            if act_parts[0]=="_msg_out" or act_parts[0]=="_msg_in":
                io_evt = {param:evt[param] for param in evt_params}
                hn_events_list.append(io_evt)
            elif agent_evt is None:
                agent_evt = {param:evt[param] for param in evt_params}
                agent_evt['activity_type'] = agent_evt['agent_id']
                hn_events_list.append(agent_evt)
    in_df = pd.DataFrame(hn_events_list)
    in_df = pm4py.format_dataframe(in_df, case_id='case_id', activity_key='activity_type', timestamp_key='timestamp')
    return pm4py.convert_to_event_log(in_df)