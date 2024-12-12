import os
import pm4py
import pandas as pd
from source.agent_miner_code import asm_sim as sim
from source.agent_miner_code import asm_stats as st
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator

def _evaluate(config,log_pm4py,stats_list,label_type='aol'):
        print(f"=== START _evaluate, {label_type}, out dir: {config.out_dir()}")

        #print("Simulating model {model_name}...") 
        #std_sim_log = sim.asm_sim(std_pn_pm4py,std_im_pm4py,std_fm_pm4py,1,max_iterations_count=100000,asm_logger=sim.asm_log_with_agent_id)
        #std_fm = std_sim_log["behavior"]["final_marking"]
        #std_sim = ('livelock' if std_sim_log['livelock'] else 'finished',str({pl:len(std_fm[pl]) for pl in std_fm}))

        pnml_file_name_base = f"soa-net-{label_type}" if config._approach=="SOA" else f"mas-net-{label_type}"
        pn_pm4py,im_pm4py,fm_pm4py = pm4py.read_pnml(os.path.join(config.out_dir(),f"{pnml_file_name_base}.pnml"))

        #gen = generalization_evaluator.apply(log_pm4py,pn_pm4py,im_pm4py,fm_pm4py)

        if config._pr_type == "pm4py":
                print(f"Calculating pm4py precision and recall for {config._run_name} - {config._config_ver} ...")
                recall = pm4py.fitness_token_based_replay(log_pm4py,pn_pm4py,im_pm4py,fm_pm4py)['log_fitness']
                print("recall = ",recall)
                prec = pm4py.precision_token_based_replay(log_pm4py,pn_pm4py,im_pm4py,fm_pm4py)
                print("precision = ",prec)
        else: # _pr_type == "entropia"
                print(f"Calculating entropy based precision and recall {config._run_name} - {config._config_ver} ...")
                label_type_parts = label_type.split("-")
                log_file_name = os.path.join(config.in_dir(),f"_log_{label_type}.xes") if len(label_type_parts)<2 else os.path.join(config.in_dir(),f"_log_{label_type_parts[1]}.xes")
                pn_file_name = os.path.join(config.out_dir(),pnml_file_name_base)+"_jbpt.pnml"
                entropia_file_name = os.path.join(config.out_dir(),pnml_file_name_base)+"_entropia.txt"
                entopia_jar = os.path.join("entropia","jbpt-pm-entropia-1.6.jar")
                os.system(f"java -Xms4G -Xmx48G -jar {entopia_jar} -empr -rel={log_file_name} -ret={pn_file_name} > {entropia_file_name}")
                prec, recall = st.read_entropia_results(entropia_file_name)

        if config._approach=="SOA":
                alg = config._soa_alg
                if alg=="DFG-WN":
                        alg_params = f"ff:{config._soa_dfg_ff}"
                elif alg=="HM":
                        alg_params = f"ff:{config._soa_hm_dt}"
                else:
                        alg_params = f"nt:{config._soa_imf_nt}"
        else:
                alg = f"a:{config._a_alg},in:{config._in_alg}"
                a_param = f"a_nt:{config._a_imf_nt}" if config._a_alg=="IMf" else f"a_ff:{config._a_dfg_ff}"
                in_param = f"in_ff:{config._in_dfg_ff}" if config._in_alg=="DFG-WN" else f"in_nt:{config._in_imf_nt}"
                alg_params = f"{a_param},{in_param}"
        dec_points = 3
        stats = {
                'config_ver' : config._config_ver,
                'approach' : config._approach,
                'alg' : str(alg),
                'alg_params' : alg_params,
                'pr_type' : config._pr_type,        
                'pl_count' : len(pn_pm4py.places),
                'tr_count' : len(pn_pm4py.transitions),
                'arc_count' : len(pn_pm4py.arcs),
                'size (pl+tr+arc)' : len(pn_pm4py.places)+len(pn_pm4py.transitions)+len(pn_pm4py.arcs),
                'recall' : round(recall,dec_points),
                'precision' : round(prec,dec_points)
                #'generalization' : round(gen,dec_points)
        }
        stats_list.append(stats)
        print(stats_list)
        stats_summary_file_name_base = config.result_summary_file_base()+f"_{label_type}"
        # _summarize(os.path.join(config.run_dir(),stats_summary_file_name_base),stats_list)

        print(f"=== END _evaluate, {label_type}, out dir: {config.out_dir()}")
        return stats_list

def _summarize(summary_file_path_base,stats_list):
        summary_df = pd.DataFrame(stats_list)
        summary_df.to_csv(f"{summary_file_path_base}.csv",index=False)
