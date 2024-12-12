import pm4py
import pandas as pd
from source.agent_miner_code import _filter as f
from source.agent_miner_code import _cluster as cl
from source.agent_miner_code import _discover as di
from source.agent_miner_code import _evaluate as ev
from source.agent_miner_code import _viz as v
import os
from source.agent_miner_code import asm_config as config

def _run_filter(example_dir=None,run_log_percent_list=[]):
    for pc in run_log_percent_list:
        filter_v = "aol"+str(pc)
        df1 = f._filter_inst_log(log_dir=example_dir,filter_ver=filter_v,case_pc=pc)
        f._discover_agent_inst_dfg(df1,log_dir=example_dir,filter_ver=filter_v)

clustering_variant = "auto_daf" # automatic clustering for direct activity flow similarity
def _run_cluster(example_dir=None,run_log_percent_list=[],
        manual_inst_cluster_map_map={},
        auto_cluster_max_dist_list=[1.0]):
#     print(manual_inst_cluster_map_map)
    for pc in run_log_percent_list:
        filter_v = "aol"+str(pc)
        for cluster_v in manual_inst_cluster_map_map:
                # print(cluster_v)
                # print(f"given: {manual_inst_cluster_map_map[cluster_v]}")                
                cl._cluster_agent_instances(log_dir=example_dir,filter_ver =filter_v,cluster_ver = cluster_v,
                inst_cluster_map=manual_inst_cluster_map_map[cluster_v])        
        for max_d in auto_cluster_max_dist_list:
                print("Discover Agent Types")
                cl._cluster_agent_instances(log_dir=example_dir,filter_ver =filter_v,cluster_ver = clustering_variant+str(int(max_d*100)),max_dfg_dist=max_d)                                        

def _run_dev_soa(run_soa_callback=None,run_name=None,example_dir=None,manual_inst_cluster_map_map={},run_log_percent_list=[100],run_cluster_max_dist_list=[1.0],run_threshold_list=[0.0],label_type='aol',more_viz=False):
        config._log_dir = example_dir
        print(f" log dir: {example_dir}")
        config._more_viz = more_viz
        config._approach = "SOA"
        config._soa_dfg_ff = 1.0
        config._run_name = f"{config._approach}_{config._soa_alg}_{run_name}"
        for pc in run_log_percent_list:
                config._filter_ver = "aol"+str(pc)
                cl_v_list = [cl_v for cl_v in manual_inst_cluster_map_map] 
                cl_v_list.extend([clustering_variant+str(int(max_d*100)) for max_d in run_cluster_max_dist_list])
                print(f"list: {cl_v_list}")
                for cl_v in cl_v_list:
                        config._cluster_ver = cl_v
                        label_type_parts = label_type.split("-")
                        print(f"config in dir: {config.in_dir()}")
                        log_file_name = os.path.join(config.in_dir(),f"_log_{label_type}.xes") if len(label_type_parts)<2 else os.path.join(config.in_dir(),f"_log_{label_type_parts[1]}.xes")
                        print(f"log file name: {log_file_name}")
                        mas_log_pm4py = pm4py.read_xes(log_file_name)
                        stats_list = []
                        for threshold in run_threshold_list:
                                config._soa_imf_nt = threshold
                                config._soa_hm_dt = threshold
                                config._soa_sm_ft = threshold
                                config._config_ver = f"{config._approach}_{config._soa_alg}{str(int(threshold*100))}"
                                if not run_soa_callback is None:
                                        run_soa_callback(config,mas_log_pm4py,stats_list,label_type=label_type,xes_log_file_name=log_file_name)

def _run_dev_am(run_am_callback=None,run_name=None,example_dir=None,manual_inst_cluster_map_map={},run_log_percent_list=[100], run_cluster_max_dist_list=[1.0],
                run_threshold_list=[(1.0,0.0)], label_type='aol', more_viz=False):
        config._log_dir = example_dir
        print(f" log dir: {example_dir}")
        config._agent_trace_id_field = "fragment_id"
        config._murata_reduction = True
        config._imf_self_loops = False
        config._more_viz = more_viz
        config._approach = "AM"
        config._run_name = f"{config._approach}_{config._a_alg}_{config._in_alg}_{run_name}"
        for pc in run_log_percent_list:
                config._filter_ver = "aol"+str(pc)
                cl_v_list = [cl_v for cl_v in manual_inst_cluster_map_map] 
                print(f"manual cluster map: {manual_inst_cluster_map_map}")
                cl_v_list.extend([clustering_variant+str(int(max_d*100)) for max_d in run_cluster_max_dist_list])
                print(f"list: {cl_v_list}")
                for cl_v in cl_v_list:
                        config._cluster_ver = cl_v
                        label_type_parts = label_type.split("-")
                        print(f"config in dir: {config.in_dir()}")
                        log_file_name = os.path.join(config.in_dir(),f"_log_{label_type}.xes") if len(label_type_parts)<2 else os.path.join(config.in_dir(),f"_log_{label_type_parts[1]}.xes")
                        print(f"log file name: {log_file_name}")
                        mas_log_pm4py = pm4py.read_xes(log_file_name)
                        stats_list = []
                        for ff,nt in run_threshold_list:
                                config._a_dfg_ff = ff
                                config._a_imf_nt = ff
                                config._a_sm_ft = ff
                                config._in_imf_nt = nt
                                config._in_sm_ft = nt
                                config._config_ver = f"{config._approach}_{config._a_alg}{str(int(ff*100))}_{config._in_alg}{str(int(nt*100))}"
                                run_am_callback(config,mas_log_pm4py,stats_list,label_type=label_type,xes_log_file_name=log_file_name)

def _run_discover_am(run_name=None,example_dir=None,manual_inst_cluster_map_map={},run_log_percent_list=[],run_cluster_max_dist_list=[],
        run_threshold_list=[(1.0,0.0)],label_type='aol',more_viz=False):
        _run_dev_am(run_am_callback=di._am_discover,run_name=run_name,example_dir=example_dir,manual_inst_cluster_map_map=manual_inst_cluster_map_map,
                run_log_percent_list=run_log_percent_list,run_cluster_max_dist_list=run_cluster_max_dist_list,
                run_threshold_list=run_threshold_list,label_type=label_type,more_viz=more_viz)
        
def _run_discover_soa(run_name=None,example_dir=None,manual_inst_cluster_map_map={},run_log_percent_list=[],run_cluster_max_dist_list=[],
        run_threshold_list=[0.0],label_type='aol',more_viz=False):
        _run_dev_soa(run_soa_callback=di._soa_discover,run_name=run_name,example_dir=example_dir,manual_inst_cluster_map_map=manual_inst_cluster_map_map,
                run_log_percent_list=run_log_percent_list,run_cluster_max_dist_list=run_cluster_max_dist_list,
                run_threshold_list=run_threshold_list,label_type=label_type,more_viz=more_viz)

def evaluate_pm4py_callback(config,mas_log_pm4py,stats_list=[],label_type='aol',xes_log_file_name=None):
        config._pr_type = "pm4py"
        stats_list = ev._evaluate(config,mas_log_pm4py,stats_list,label_type=label_type)
        return stats_list

def evaluate_entropia_callback(config,mas_log_pm4py,stats_list=[],label_type='aol',xes_log_file_name=None):
        config._pr_type = "entropia"
        stats_list = ev._evaluate(config,mas_log_pm4py,stats_list,label_type=label_type)
        return stats_list

def _run_eval_soa_pm4py(run_name=None,example_dir=None,manual_inst_cluster_map_map={},run_log_percent_list=[],run_cluster_max_dist_list=[],
        run_threshold_list=[0.0],label_type='aol',more_viz=False):
        _run_dev_soa(run_soa_callback=di._soa_discover,run_name=run_name,example_dir=example_dir,manual_inst_cluster_map_map=manual_inst_cluster_map_map,
                run_log_percent_list=run_log_percent_list,run_cluster_max_dist_list=run_cluster_max_dist_list,
                run_threshold_list=run_threshold_list,label_type=label_type)

def _run_eval_am_pm4py(run_name=None,example_dir=None, manual_inst_cluster_map_map={},run_log_percent_list=[],run_cluster_max_dist_list=[],
        run_threshold_list=[(1.0,0.0)],label_type='aol',more_viz=False):
        _run_dev_am(run_am_callback=evaluate_pm4py_callback,run_name=run_name,example_dir=example_dir,manual_inst_cluster_map_map=manual_inst_cluster_map_map,
                run_log_percent_list=run_log_percent_list,run_cluster_max_dist_list=run_cluster_max_dist_list,
                run_threshold_list=run_threshold_list,label_type=label_type)

def _run_eval_soa_entropia(run_name=None,example_dir=None,manual_inst_cluster_map_map={},run_log_percent_list=[],run_cluster_max_dist_list=[],
        run_threshold_list=[0.0],label_type='aol',more_viz=False):
        _run_dev_soa(run_soa_callback=evaluate_entropia_callback,run_name=run_name,example_dir=example_dir,manual_inst_cluster_map_map=manual_inst_cluster_map_map,
                run_log_percent_list=run_log_percent_list,run_cluster_max_dist_list=run_cluster_max_dist_list,
                run_threshold_list=run_threshold_list,label_type=label_type)

def _run_eval_am_entropia(run_name=None,example_dir=None, manual_inst_cluster_map_map={},run_log_percent_list=[],run_cluster_max_dist_list=[],
        run_threshold_list=[(1.0,0.0)],label_type='aol',more_viz=False):
        _run_dev_am(run_am_callback=evaluate_entropia_callback,run_name=run_name,example_dir=example_dir, manual_inst_cluster_map_map=manual_inst_cluster_map_map,
                run_log_percent_list=run_log_percent_list,run_cluster_max_dist_list=run_cluster_max_dist_list,
                run_threshold_list=run_threshold_list,label_type=label_type)

def _run_viz_soa(run_name=None,example_dir=None,manual_inst_cluster_map_map={},run_log_percent_list=[],run_cluster_max_dist_list=[],
        run_threshold_list=[0.0],label_type='aol',more_viz=False):
        _run_dev_soa(run_soa_callback=v._visualize_soa,run_name=run_name,example_dir=example_dir,manual_inst_cluster_map_map=manual_inst_cluster_map_map,
                run_log_percent_list=run_log_percent_list,run_cluster_max_dist_list=run_cluster_max_dist_list,
                run_threshold_list=run_threshold_list,label_type=label_type)

def _run_viz_am(run_name=None,example_dir=None, manual_inst_cluster_map_map={},run_log_percent_list=[],run_cluster_max_dist_list=[],
        run_threshold_list=[(1.0,0.0)],label_type='aol',more_viz=False):
        _run_dev_am(run_am_callback=v._visualize_am,run_name=run_name,example_dir=example_dir, manual_inst_cluster_map_map=manual_inst_cluster_map_map,
                run_log_percent_list=run_log_percent_list,run_cluster_max_dist_list=run_cluster_max_dist_list,
                run_threshold_list=run_threshold_list,label_type=label_type)

def _run_completed(run_name="run"):
    marker_file_name = f"_completed_{run_name}.txt"
    with open(marker_file_name,"w") as log_file:
        print(f"Run '{run_name}' completed.",file=log_file)