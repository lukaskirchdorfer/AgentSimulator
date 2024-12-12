import pandas as pd
import pm4py
import os
import datetime
from source.agent_miner_code import asm_data as da
from source.agent_miner_code import asm_dfg as dfg
from source.agent_miner_code import asm_pn as pn
from pm4py.objects.petri_net.utils import petri_utils

def discover_agent_model(config,agent_id,agent_log_pm4py,agent_xes_log_name=None):
        a_dfg_obj = dfg.discover_dfg_pm4py(agent_id,agent_log_pm4py,activity_frequency_filter=config._a_dfg_ff)
        if config._a_alg=="DFG-WN":
                a_pn_pm4py, a_im_pm4py, a_fm_pm4py = pn.discover_pn_from_dfg(a_dfg_obj)
        elif config._a_alg=="SM":
                pm4py.write_xes(agent_log_pm4py,agent_xes_log_name)
                a_pn_pm4py, a_im_pm4py, a_fm_pm4py = pn.discover_pn_sm_pm4py(agent_xes_log_name,frequency_threshold=config._a_sm_ft)
        else:
                a_pn_pm4py, a_im_pm4py, a_fm_pm4py = pn.discover_pn_imf_pm4py(agent_log_pm4py,config._a_alg,noise_threshold=config._a_imf_nt)
                #a_pn_pm4py, a_im_pm4py, a_fm_pm4py = pm4py.discover_petri_net_inductive(agent_log_pm4py,noise_threshold=config._a_imf_nt)
        return (a_pn_pm4py, a_im_pm4py, a_fm_pm4py,a_dfg_obj)

def _am_discover(config,mas_log_pm4py,stats_list,label_type='aol',xes_log_file_name=None):
        event_concept_name_attr = 'activity_type' if label_type=='aol' else 'agent_activity_type'
        print(f"=== START _am_discover, out_dir: {config.out_dir()}")
        os.system(f"mkdir {config.run_dir()}")
        os.system(f"mkdir {config.out_dir()}")

        print(datetime.datetime.now(),"Discover ASM agent models...")
        agents_out_dir = os.path.join(config.out_dir(),"agents")
        os.system(f"mkdir {agents_out_dir}")
        agent_model_map = {}
        agent_stats_list = []
        mas_log_df = pm4py.convert_to_dataframe(mas_log_pm4py)
        log_groupby_agent = mas_log_df.groupby(['agent_id'],sort=False)
        for agent_id,agent_log_df in log_groupby_agent:
                print(f"{datetime.datetime.now()} Discovering agent net for {str(agent_id)} ...")
                df0 = pm4py.format_dataframe(agent_log_df, case_id=config._agent_trace_id_field, activity_key=event_concept_name_attr, timestamp_key='timestamp')

                # save agent logs
                df0.to_csv(os.path.join(agents_out_dir,str(agent_id)+'_log.csv'))

                agent_log_pm4py = pm4py.convert_to_event_log(df0)
                agent_file_name = os.path.join(agents_out_dir,f"{agent_id}_log.xes")
                cl_pn_pm4py, cl_im_pm4py, cl_fm_pm4py, cl_dfg_obj = discover_agent_model(config,agent_id,agent_log_pm4py,agent_xes_log_name=agent_file_name)
                if config._murata_reduction:
                        if config._more_viz:
                                pn.viz_pn(cl_pn_pm4py, cl_im_pm4py, cl_fm_pm4py,os.path.join(agents_out_dir,str(agent_id)+f'_agent-net-{label_type}_0'))
                        cl_pn_pm4py, cl_im_pm4py, cl_fm_pm4py = pn.apply_murata_reduction_rules(cl_pn_pm4py, cl_im_pm4py, cl_fm_pm4py)
                agent_model_map[agent_id] = (cl_pn_pm4py, cl_im_pm4py, cl_fm_pm4py,cl_dfg_obj)
                pn.write_pnml(cl_pn_pm4py, cl_im_pm4py, cl_fm_pm4py,os.path.join(agents_out_dir,str(agent_id)+f'_agent-net-{label_type}'))
                if config._more_viz:
                        dfg.viz_dfg(cl_dfg_obj,os.path.join(agents_out_dir,str(agent_id)+f'_dfg_{label_type}'))
                        pn.viz_pn(cl_pn_pm4py, cl_im_pm4py, cl_fm_pm4py,os.path.join(agents_out_dir,str(agent_id)+f'_agent-net-{label_type}'))
                        agent_stats_list.append({
                                'agent_type' : agent_id,
                                'pl_count' : len(cl_pn_pm4py.places),
                                'tr_count' : len(cl_pn_pm4py.transitions),
                                '(pl+tr)_count' : len(cl_pn_pm4py.places)+len(cl_pn_pm4py.transitions),
                                'arc_count' : len(cl_pn_pm4py.arcs),
                                '(pl+tr+arc)_count' : len(cl_pn_pm4py.places)+len(cl_pn_pm4py.transitions)+len(cl_pn_pm4py.arcs)
                })
        if config._more_viz:
                agent_stats_df = pd.DataFrame(agent_stats_list)
                agent_stats_df.to_csv(os.path.join(agents_out_dir,f'agents_pn_summary_{label_type}.csv'))

        print(datetime.datetime.now(),"Creating the i-net log ...") 
        in_log_pm4py, in_log_df = da.create_interaction_log(mas_log_df,agent_trace_col_name=config._agent_trace_id_field)
        in_log_name = os.path.join(config.out_dir(),"_log_in.xes")
        pm4py.write_xes(in_log_pm4py,in_log_name)

        in_log_df_name = os.path.join(config.out_dir(),"_log_in.csv")
        in_log_df.to_csv(in_log_df_name)

        print(datetime.datetime.now(),f"Discovering interaction net with {config._in_alg} ...") 
        if config._in_alg=="DFG-WN":
                in_dfg_obj = dfg.discover_dfg_pm4py("i-net",in_log_pm4py,activity_frequency_filter=config._in_dfg_ff)
                in_pm4py,in_im_pm4py,in_fm_pm4py = pn.discover_pn_from_dfg(in_dfg_obj)
        elif config._in_alg=="SM":
                in_pm4py,in_im_pm4py,in_fm_pm4py = pn.discover_pn_sm_pm4py(in_log_name,frequency_threshold=config._in_sm_ft)
        else:
                in_pm4py,in_im_pm4py,in_fm_pm4py = pn.discover_pn_imf_pm4py(in_log_pm4py,config._in_alg,noise_threshold=config._in_imf_nt)
                #in_pm4py,in_im_pm4py,in_fm_pm4py = pm4py.discover_petri_net_inductive(in_log_pm4py,noise_threshold=config._in_imf_nt)
                if not config._imf_self_loops:
                        pn.remove_agent_self_loops_in(in_pm4py)
        if config._murata_reduction:
                if config._more_viz:
                        pn.viz_pn(in_pm4py,in_im_pm4py,in_fm_pm4py,os.path.join(config.out_dir(),f"i-net-{label_type}_0"))
                in_pm4py,in_im_pm4py,in_fm_pm4py = pn.apply_murata_reduction_rules(in_pm4py,in_im_pm4py,in_fm_pm4py)
        pn.write_pnml(in_pm4py,in_im_pm4py,in_fm_pm4py,os.path.join(config.out_dir(),f"i-net-{label_type}"))
        if config._more_viz:
                da.save_xes_log(pm4py.convert_to_dataframe(in_log_pm4py),os.path.join(config.out_dir(),f"i-net-{label_type}_log"),save_xes=False)
                pn.viz_pn(in_pm4py,in_im_pm4py,in_fm_pm4py,os.path.join(config.out_dir(),f"i-net-{label_type}"))
        asm_interaction_model = (in_pm4py,in_im_pm4py,in_fm_pm4py)

        print(datetime.datetime.now(),"Discovering MAS net...") 
        mas_pn_pm4py,mas_pn_im_pm4py,mas_pn_fm_pm4py = pn.integrate_mas_pn_pm4py(in_pm4py,in_im_pm4py,in_fm_pm4py,agent_model_map)
        if config._murata_reduction:
                if config._more_viz:
                        pn.viz_pn(mas_pn_pm4py,mas_pn_im_pm4py,mas_pn_fm_pm4py,os.path.join(config.out_dir(),f"mas-net-{label_type}_0"))
                mas_pn_pm4py,mas_pn_im_pm4py,mas_pn_fm_pm4py = pn.apply_murata_reduction_rules(mas_pn_pm4py,mas_pn_im_pm4py,mas_pn_fm_pm4py)
        pn.write_pnml(mas_pn_pm4py,mas_pn_im_pm4py,mas_pn_fm_pm4py,os.path.join(config.out_dir(),f"mas-net-{label_type}"))
        asm_i_mas_model= (mas_pn_pm4py,mas_pn_im_pm4py,mas_pn_fm_pm4py)

        aal_aol = False
        if label_type=="aal":
                # update transition labels to activity-only
                mas_pn_pm4py2 = petri_utils.merge(None,[mas_pn_pm4py])
                for tr in mas_pn_pm4py2.transitions:
                        if not tr.label is None:
                                parts = tr.label.split("|")
                                if len(parts)>1:
                                        aal_aol = True
                                        tr.label = parts[1]
                pn.write_pnml(mas_pn_pm4py2,mas_pn_im_pm4py,mas_pn_fm_pm4py,os.path.join(config.out_dir(),f"mas-net-aal-aol"))
        # print("Net:", mas_pn_pm4py2)
        # print("Initial marking:", mas_pn_im_pm4py)
        # print("Final marking:", mas_pn_fm_pm4py)

        if config._more_viz:
                pn.viz_pn(mas_pn_pm4py,mas_pn_im_pm4py,mas_pn_fm_pm4py,os.path.join(config.out_dir(),f"mas-net-{label_type}"))
                if aal_aol:
                        pn.viz_pn(mas_pn_pm4py2,mas_pn_im_pm4py,mas_pn_fm_pm4py,os.path.join(config.out_dir(),f"mas-net-aal-aol"))

        print(f"=== END of _am_discover, out dir: {config.out_dir()}")
        return (agent_model_map,asm_interaction_model,asm_i_mas_model)

def _am_discover2(config,mas_log_pm4py,stats_list):
        print(f"=== START _am_discover2, out_dir: {config.out_dir()}")
        os.system(f"mkdir {config.run_dir()}")
        os.system(f"mkdir {config.out_dir()}")

        mas_log_with_msg_df = da.add_message_events_to_log_with_fragments_df(mas_log_pm4py)

        print(datetime.datetime.now(),"Discover ASM agent models...")
        agents_out_dir = os.path.join(config.out_dir(),"agents")
        os.system(f"mkdir {agents_out_dir}")
        agent_model_map = {}
        agent_stats_list = []
        log_groupby_agent = mas_log_with_msg_df.groupby(['agent_id'],sort=False)
        for agent_id,agent_log_df in log_groupby_agent:
                print(f"{datetime.datetime.now()} Discovering extended agent net for {str(agent_id)} ...")
                df0 = pm4py.format_dataframe(agent_log_df, case_id=config._agent_trace_id_field, activity_key='activity_type', timestamp_key='timestamp')
                agent_log_pm4py = pm4py.convert_to_event_log(df0)
                cl_pn_pm4py, cl_im_pm4py, cl_fm_pm4py, cl_dfg_obj = discover_agent_model(config,agent_id,agent_log_pm4py)
                agent_model_map[agent_id] = (cl_pn_pm4py, cl_im_pm4py, cl_fm_pm4py,cl_dfg_obj)
                cl_pn_pm4py, cl_im_pm4py, cl_fm_pm4py = pn.apply_murata_reduction_rules(cl_pn_pm4py, cl_im_pm4py, cl_fm_pm4py)
                if config._more_viz:
                        dfg.viz_dfg(cl_dfg_obj,os.path.join(agents_out_dir,str(agent_id)+'_dfg'))
                        pn.viz_pn(cl_pn_pm4py, cl_im_pm4py, cl_fm_pm4py,os.path.join(agents_out_dir,str(agent_id)+'_agent-net_wf'))
                # all agent nets ( as WF nets) are assumed to have one start (source) place
                a_surce_pl = [pl for pl in cl_pn_pm4py.places if pl in cl_im_pm4py][0]
                petri_utils.remove_place(cl_pn_pm4py,a_surce_pl)
                # all cluster nets ( as WF nets) are assumed to have one end (sink) place
                a_sink_pl = [pl for pl in cl_pn_pm4py.places if pl in cl_fm_pm4py][0]
                petri_utils.remove_place(cl_pn_pm4py,a_sink_pl)
                pn.write_pnml(cl_pn_pm4py, cl_im_pm4py, cl_fm_pm4py,os.path.join(agents_out_dir,str(agent_id)+'_agent-net'))
                if config._more_viz:
                        pn.viz_pn(cl_pn_pm4py, cl_im_pm4py, cl_fm_pm4py,os.path.join(agents_out_dir,str(agent_id)+'_agent-net'))
                        agent_stats_list.append({
                                'agent_type' : agent_id,
                                'pl_count' : len(cl_pn_pm4py.places),
                                'tr_count' : len(cl_pn_pm4py.transitions),
                                '(pl+tr)_count' : len(cl_pn_pm4py.places)+len(cl_pn_pm4py.transitions),
                                'arc_count' : len(cl_pn_pm4py.arcs),
                                '(pl+tr+arc)_count' : len(cl_pn_pm4py.places)+len(cl_pn_pm4py.transitions)+len(cl_pn_pm4py.arcs)
                })
        if config._more_viz:
                agent_stats_df = pd.DataFrame(agent_stats_list)
                agent_stats_df.to_csv(os.path.join(agents_out_dir,'agents_pn_summary.csv'))

        print(datetime.datetime.now(),"Discovering MAS net...") 
        mas_pn_pm4py,mas_pn_im_pm4py,mas_pn_fm_pm4py = pn.integrate_mas_pn2_pm4py(agent_model_map)
        if config._murata_reduction:
                if config._more_viz:
                        pn.viz_pn(mas_pn_pm4py,mas_pn_im_pm4py,mas_pn_fm_pm4py,os.path.join(config.out_dir(),"mas-net_0"))
                mas_pn_pm4py,mas_pn_im_pm4py,mas_pn_fm_pm4py = pn.apply_murata_reduction_rules(mas_pn_pm4py,mas_pn_im_pm4py,mas_pn_fm_pm4py)
        pn.write_pnml(mas_pn_pm4py,mas_pn_im_pm4py,mas_pn_fm_pm4py,os.path.join(config.out_dir(),"mas-net"))
        if config._more_viz:
                pn.viz_pn(mas_pn_pm4py,mas_pn_im_pm4py,mas_pn_fm_pm4py,os.path.join(config.out_dir(),"mas-net"))
        mas_model= (mas_pn_pm4py,mas_pn_im_pm4py,mas_pn_fm_pm4py)

        print(f"=== END of _am_discover2, out dir: {config.out_dir()}")
        return (agent_model_map,agent_model_map['_env'],mas_model)#asm_i_mas_model)

def _am_discover3(config,agent_trace_set_pm4py,stats_list):
        print(f"=== START _am_discover3, out_dir: {config.out_dir()}")
        os.system(f"mkdir {config.run_dir()}")
        os.system(f"mkdir {config.out_dir()}")

        print(datetime.datetime.now(),"Creating the io-net log ...") 
        io_log_pm4py = da.create_io_log(agent_trace_set_pm4py,agent_trace_col_name=config._agent_trace_id_field)

        print(datetime.datetime.now(),f"Discovering io net with {config._in_alg} ...") 
        if config._in_alg=="DFG-WN":
                in_dfg_obj = dfg.discover_dfg_pm4py("io-net",io_log_pm4py,activity_frequency_filter=config._in_dfg_ff)
                ion_pm4py,ion_im_pm4py,ion_fm_pm4py = pn.discover_pn_from_dfg(in_dfg_obj)
                pm4py.di
        else:
                ion_pm4py,ion_im_pm4py,ion_fm_pm4py = pn.discover_pn_imf_pm4py(io_log_pm4py,config._in_alg,noise_threshold=config._in_imf_nt)
                #in_pm4py,in_im_pm4py,in_fm_pm4py = pm4py.discover_petri_net_inductive(in_log_pm4py,noise_threshold=config._in_imf_nt)
                #if not config._imf_self_loops:
                #        pn.remove_agent_self_loops_in(in_pm4py)
        if config._murata_reduction:
                if config._more_viz:
                        pn.viz_pn(ion_pm4py,ion_im_pm4py,ion_fm_pm4py,os.path.join(config.out_dir(),"i-net_0"))
                ion_pm4py,ion_im_pm4py,ion_fm_pm4py = pn.apply_murata_reduction_rules(ion_pm4py,ion_im_pm4py,ion_fm_pm4py)
        pn.write_pnml(ion_pm4py,ion_im_pm4py,ion_fm_pm4py,os.path.join(config.out_dir(),"i-net"))
        if config._more_viz:
                da.save_xes_log(pm4py.convert_to_dataframe(io_log_pm4py),os.path.join(config.out_dir(),"i-net_log"),save_xes=False)
                pn.viz_pn(ion_pm4py,ion_im_pm4py,ion_fm_pm4py,os.path.join(config.out_dir(),"i-net"))
        asm_interaction_model = (ion_pm4py,ion_im_pm4py,ion_fm_pm4py)

        mas_log_with_msg_df = da.add_message_events_to_log_with_fragments_df(agent_trace_set_pm4py,ion_pm4py)

        print(datetime.datetime.now(),"Discover ASM agent models...")
        agents_out_dir = os.path.join(config.out_dir(),"agents")
        os.system(f"mkdir {agents_out_dir}")
        agent_model_map = {}
        agent_dfg_obj_map = {}
        agent_stats_list = []
        log_groupby_agent = mas_log_with_msg_df.groupby(['agent_id'],sort=False)
        for agent_id,agent_log_df in log_groupby_agent:
                print(f"{datetime.datetime.now()} Discovering agent net for {str(agent_id)} ...")
                df0 = pm4py.format_dataframe(agent_log_df, case_id=config._agent_trace_id_field, activity_key='activity_type', timestamp_key='timestamp')
                agent_log_pm4py = pm4py.convert_to_event_log(df0)
                cl_pn_pm4py, cl_im_pm4py, cl_fm_pm4py, cl_dfg_obj = discover_agent_model(config,agent_id,agent_log_pm4py)
                if config._murata_reduction:
                        if config._more_viz:
                                pn.viz_pn(cl_pn_pm4py, cl_im_pm4py, cl_fm_pm4py,os.path.join(agents_out_dir,str(agent_id)+'_agent-net_0'))
                        cl_pn_pm4py, cl_im_pm4py, cl_fm_pm4py = pn.apply_murata_reduction_rules(cl_pn_pm4py, cl_im_pm4py, cl_fm_pm4py)
                agent_model_map[agent_id] = (cl_pn_pm4py, cl_im_pm4py, cl_fm_pm4py,cl_dfg_obj)
                agent_dfg_obj_map[agent_id] = cl_dfg_obj
                pn.write_pnml(cl_pn_pm4py, cl_im_pm4py, cl_fm_pm4py,os.path.join(agents_out_dir,str(agent_id)+'_agent-net'))
                if config._more_viz:
                        dfg.viz_dfg(cl_dfg_obj,os.path.join(agents_out_dir,str(agent_id)+'_dfg'))
                        pn.viz_pn(cl_pn_pm4py, cl_im_pm4py, cl_fm_pm4py,os.path.join(agents_out_dir,str(agent_id)+'_agent-net'))
                        agent_stats_list.append({
                                'agent_type' : agent_id,
                                'pl_count' : len(cl_pn_pm4py.places),
                                'tr_count' : len(cl_pn_pm4py.transitions),
                                '(pl+tr)_count' : len(cl_pn_pm4py.places)+len(cl_pn_pm4py.transitions),
                                'arc_count' : len(cl_pn_pm4py.arcs),
                                '(pl+tr+arc)_count' : len(cl_pn_pm4py.places)+len(cl_pn_pm4py.transitions)+len(cl_pn_pm4py.arcs)
                })
        if config._more_viz:
                agent_stats_df = pd.DataFrame(agent_stats_list)
                agent_stats_df.to_csv(os.path.join(agents_out_dir,'agents_pn_summary.csv'))

        print(datetime.datetime.now(),"Discovering MAS net...") 
        mas_pn_pm4py,mas_pn_im_pm4py,mas_pn_fm_pm4py = pn.integrate_mas_pn3_pm4py(ion_pm4py,ion_im_pm4py,ion_fm_pm4py,agent_model_map)
        if config._more_viz:
                pn.viz_pn(mas_pn_pm4py,mas_pn_im_pm4py,mas_pn_fm_pm4py,os.path.join(config.out_dir(),"mas-net-with-io"))
        for tr in mas_pn_pm4py.transitions:
                if (not tr.label is None) and (len(tr.label.split("|"))>1):
                        tr.label = None
        if config._murata_reduction:
                if config._more_viz:
                        pn.viz_pn(mas_pn_pm4py,mas_pn_im_pm4py,mas_pn_fm_pm4py,os.path.join(config.out_dir(),"mas-net_0"))
                mas_pn_pm4py,mas_pn_im_pm4py,mas_pn_fm_pm4py = pn.apply_murata_reduction_rules(mas_pn_pm4py,mas_pn_im_pm4py,mas_pn_fm_pm4py)
        pn.write_pnml(mas_pn_pm4py,mas_pn_im_pm4py,mas_pn_fm_pm4py,os.path.join(config.out_dir(),"mas-net"))
        if config._more_viz:
                pn.viz_pn(mas_pn_pm4py,mas_pn_im_pm4py,mas_pn_fm_pm4py,os.path.join(config.out_dir(),"mas-net"))
        asm_i_mas_model= (mas_pn_pm4py,mas_pn_im_pm4py,mas_pn_fm_pm4py)

        print(f"=== END of _am_discover, out dir: {config.out_dir()}")
        return (agent_model_map,asm_interaction_model,asm_i_mas_model)

def _soa_discover(config,mas_log_pm4py,stats_list,label_type='aol',xes_log_file_name=None):
        print(f"=== START _soa_discover, {label_type}, out dir:{config.out_dir()}")
        os.system(f"mkdir {config.run_dir()}")
        os.system(f"mkdir {config.out_dir()}")

        std_dfg_obj = dfg.discover_dfg_pm4py("soa",mas_log_pm4py,activity_frequency_filter=config._soa_dfg_ff)
        if config._soa_alg=="DFG-WN":
                std_pn_pm4py, std_im_pm4py, std_fm_pm4py = pn.discover_pn_from_dfg(std_dfg_obj)
        elif config._soa_alg=="HM":
                std_pn_pm4py, std_im_pm4py, std_fm_pm4py = pn.discover_pn_hm_pm4py(mas_log_pm4py,dependency_threshold=config._soa_hm_dt)
        elif config._soa_alg=="ALPHA":
                std_pn_pm4py, std_im_pm4py, std_fm_pm4py = pn.discover_pn_alpha_pm4py(mas_log_pm4py)
        elif config._soa_alg=="SM":
                std_pn_pm4py, std_im_pm4py, std_fm_pm4py = pn.discover_pn_sm_pm4py(xes_log_file_name,frequency_threshold=config._soa_sm_ft)
        else: #config._soa_alg=="IMf"
                std_pn_pm4py, std_im_pm4py, std_fm_pm4py = pn.discover_pn_imf_pm4py(mas_log_pm4py,config._soa_alg,noise_threshold=config._soa_imf_nt)
                #std_pn_pm4py, std_im_pm4py, std_fm_pm4py = pm4py.discover_petri_net_inductive(log_pm4py,noise_threshold=config._soa_imf_nt)

        pn.write_pnml(std_pn_pm4py,std_im_pm4py,std_fm_pm4py,os.path.join(config.out_dir(),f"soa-net-{label_type}"))
        aal_aol = False
        if label_type=="aal-aol":
                # update transition labels to activity-only
                std_pn_pm4py2 = petri_utils.merge(None,[std_pn_pm4py])
                for tr in std_pn_pm4py2.transitions:
                        if not tr.label is None:
                                parts = tr.label.split("|")
                                if len(parts)>1:
                                        aal_aol = True
                                        tr.label = parts[1]
                pn.write_pnml(std_pn_pm4py2,std_im_pm4py,std_fm_pm4py,os.path.join(config.out_dir(),f"soa-net-aal-aol"))

        if config._more_viz:
                dfg.viz_dfg(std_dfg_obj,os.path.join(config.out_dir(), f"soa_dfg_{label_type}"),more_viz=True)
                pn.viz_pn(std_pn_pm4py,std_im_pm4py,std_fm_pm4py,os.path.join(config.out_dir(),f"soa-net-{label_type}"))
                if aal_aol:
                        pn.viz_pn(std_pn_pm4py2,std_im_pm4py,std_fm_pm4py,os.path.join(config.out_dir(),f"soa-net-aal-aol"))

        print(f"=== END of _soa_discover, {label_type}, out dir: {config.out_dir()}")
        return (std_pn_pm4py,std_im_pm4py,std_fm_pm4py)