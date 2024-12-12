import os
import datetime
import networkx as nx
import asm_pn as pn
import pandas as pd
import asm_sim as sim

_example_name = "paper_example2"
_seed_count = 3
_required_case_count_list = [2,8,32,128]

print(datetime.datetime.now(),f"================= START of main_generate_example_log ({_example_name}) ==================")
print("current folder:",os.getcwd())
example_dir = os.path.join(os.getcwd(),_example_name)
source_pn_file_name_base = os.path.join(example_dir,"_source_"+_example_name)
source_pn_nx = nx.read_gml(source_pn_file_name_base+".gml")
source_pn_pm4py, source_pn_im_pm4py, source_pn_fm_pm4py = pn.pn_nx2pm4py(source_pn_nx)
pn_file_name_base = os.path.join(example_dir,"_source"+_example_name+"_mas_pn")
pn.write_pnml(source_pn_pm4py, source_pn_im_pm4py, source_pn_fm_pm4py,pn_file_name_base)
pn.viz_pn(source_pn_pm4py,source_pn_im_pm4py,source_pn_fm_pm4py,pn_file_name_base)
for req_case_count in _required_case_count_list:
        gen_log_dir = os.path.join(example_dir,f"{_example_name}_{req_case_count}cases") 
        os.system(f"rmdir /q /s {gen_log_dir}")
        os.system(f"mkdir {gen_log_dir}")
        for seed in range(1,_seed_count+1):
                input_log_df = pd.DataFrame()
                print(f"Simulating {req_case_count} cases ...")
                sim_log = sim.asm_sim(source_pn_pm4py, source_pn_im_pm4py, source_pn_fm_pm4py,req_case_count,max_iterations_count=10000,asm_logger = sim.asm_log_with_agent_id)
                input_log_df = input_log_df.append(sim_log["behavior"]["log_rows"])
                input_log_df.to_csv(os.path.join(gen_log_dir,f"{_example_name}_{str(seed)}.csv"),index=False)
        print(datetime.datetime.now(),f"================= END of main_generate_example_log ({_example_name}) ==================")
