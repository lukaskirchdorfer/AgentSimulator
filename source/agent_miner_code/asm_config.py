import os
import pm4py

_log_dir = None
_filter_ver = None
_cluster_ver= None
_config_ver = None
_run_name = None

# discover params
_agent_trace_id_field = "fragment_id" # fragment is an agent trace, fragment_id is an agent trace id
_approach = "AM"
_soa_alg = pm4py.algo.discovery.inductive.algorithm.Variants.IMf
_soa_dfg_ff = 1.0
_soa_imf_nt = 0.0
_soa_hm_dt = 0.0
_soa_sm_ft = 0.0
_in_alg = pm4py.algo.discovery.inductive.algorithm.Variants.IMf
_in_dfg_ff = 1.0
_in_imf_nt = 0.0
_in_sm_ft = 0.0
_a_alg = "DFG-WN"
_a_dfg_ff = 1.0
_a_imf_nt = 0.0
_a_sm_ft = 0.0
_murata_reduction = True
_imf_self_loops = False
_more_viz = False
# evaluate params
_pr_type = "pm4py" # or "entropia" 

def in_dir():
    print(f"current wd: {os.getcwd()}")
    return os.path.join(os.getcwd(),_log_dir,_filter_ver,'_no_clustering_') if _cluster_ver is None else os.path.join(os.getcwd(),_log_dir,_filter_ver,_cluster_ver)

def run_dir():
    return os.path.join(in_dir(),_run_name)

def result_summary_file_base():
    return os.path.join(run_dir(),f"_stats__{_log_dir}__{_filter_ver}__{_cluster_ver}__{_run_name}__{_pr_type}")    

def out_dir():
    return os.path.join(run_dir(),_config_ver)
