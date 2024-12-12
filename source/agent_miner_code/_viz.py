import pm4py
import os
from source.agent_miner_code import asm_pn as pn
from source.agent_miner_code import asm_cluster as cl

def _visualize_am(config,mas_log_pm4py,stats_list,label_type='aol',xes_log_file_name=None):
        print(f"=== visualize_am, {label_type}, out dir: {config.out_dir()}")
        # Visualize i-net
        in_pm4py,in_im_pm4py,in_fm_pm4py = pm4py.read_pnml(os.path.join(config.out_dir(),f"i-net-{label_type}.pnml"))
        pn.viz_pn(in_pm4py,in_im_pm4py,in_fm_pm4py,os.path.join(config.out_dir(),f"i-net-{label_type}_viz"))

        # Visualize agent nets
        agents_out_dir = os.path.join(config.out_dir(),"agents")
        agent_cluster_inst_map = cl.read_cluster_instance_map_csv(os.path.join(config.in_dir(),"cluster_inst_map.csv"))
        for cluster_id in agent_cluster_inst_map:
                cluster_file_path_base = os.path.join(agents_out_dir,str(f"('{cluster_id}',)")+f"_agent-net-{label_type}")
                print(f"cluster file base path: {cluster_file_path_base}")
                cl_pm4py,cl_im_pm4py,cl_fm_pm4py = pm4py.read_pnml(cluster_file_path_base+'.pnml')
                pn.viz_pn(cl_pm4py,cl_im_pm4py,cl_fm_pm4py,cluster_file_path_base+'_viz')

        # Visualize mas net
        mas_pm4py,mas_im_pm4py,mas_fm_pm4py = pm4py.read_pnml(os.path.join(config.out_dir(),f"mas-net-{label_type}.pnml"))
        #for tr in mas_pm4py.transitions:
        #        if not tr.label is None:
        #                tr.label = tr.name
        pn.viz_pn(mas_pm4py,mas_im_pm4py,mas_fm_pm4py,os.path.join(config.out_dir(),f"mas-net-{label_type}_viz"))

def _visualize_soa(config,mas_log_pm4py,stats_list,label_type='aol',xes_log_file_name=None):
        print(f"=== visualize_soa, {label_type}, out dir: {config.out_dir()}")
        # Visualize mas net
        mas_pm4py,mas_im_pm4py,mas_fm_pm4py = pm4py.read_pnml(os.path.join(config.out_dir(),f"soa-net-{label_type}.pnml"))
        pn.viz_pn(mas_pm4py,mas_im_pm4py,mas_fm_pm4py,os.path.join(config.out_dir(),f"soa-net-{label_type}_viz"))
