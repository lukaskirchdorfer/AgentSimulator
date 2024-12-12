import random as rn
import snakes.plugins
import datetime as dt
from pm4py.objects.petri_net.utils import petri_utils
sn = snakes.plugins.load("gv", "snakes.nets", "sn")

#USED
def pm4py2sn(pn_pm4py,pn_im_pm4py,token_value='case'):
    pnSn = sn.PetriNet(pn_pm4py.name)
    for pl in pn_pm4py.places:
        marking = sn.MultiSet([sn.Token(token_value) for i in range(pn_im_pm4py[pl])])
        pnSn.add_place(sn.Place(pl.name,marking))
    for tr in pn_pm4py.transitions:
        pnSn.add_transition(sn.Transition(tr.name)) 
    varName = "c"
    for arc in pn_pm4py.arcs:
        tr = petri_utils.get_transition_by_name(pn_pm4py,arc.source.name)
        if not tr is None:
            pnSn.add_output(arc.target.name,tr.name,sn.Variable(varName))
        else:
            tr = arc.target
            pnSn.add_input(arc.source.name,tr.name,sn.Variable(varName))
    return pnSn

#USED
def asm_log_with_agent_id(pn_pm4py, selected_transition, selected_binding):
    transition_id = selected_transition.name
    tr_pm4py = petri_utils.get_transition_by_name(pn_pm4py,transition_id)
    if tr_pm4py.label is None:
        return None
    else:
        label_parts = tr_pm4py.name.split('+')
        agent_id = label_parts[0].strip()
        activity_label = tr_pm4py.label 
        case_id = selected_binding["c"]
        finish_timestamp = asm_log_with_agent_id.timestamp
        asm_log_with_agent_id.timestamp += dt.timedelta(minutes=1)
        return {
            'time':finish_timestamp,
            'activity':activity_label,
            'case':case_id,
            'resource':agent_id,
            'transition_id':transition_id
        }
asm_log_with_agent_id.timestamp = dt.datetime.now()
asm_log_with_agent_id.header = "time:timestamp,concept:name,case:concept:name,org:resource,transition_id"

#USED
def asm_sim(pn_pm4py,pn_im_pm4py, pn_fm_pm4py,case_count,max_iterations_count = None,asm_logger = asm_log_with_agent_id):
    def is_final_marking_reached(pn_sn,fm_pm4py):
        cur_sim_marking = pn_sn.get_marking()
        for final_pl in fm_pm4py:
            final_pl_token_count = fm_pm4py[final_pl]
            if (not final_pl.name in cur_sim_marking):
                return False
            elif len(list(cur_sim_marking[final_pl.name])) < final_pl_token_count:
                return False
        return True
    def get_pn_idx_for_sim_step(pn_sn_list):
        idx = int(rn.uniform(0,len(pn_sn_list)))
        return idx

    pn_sn_list = []
    for i in range(1,case_count+1): 
        token_value = f"case{i}"
        pn_sn_list.append(pm4py2sn(pn_pm4py,pn_im_pm4py,token_value))
    if max_iterations_count is None:
        max_iterations_count = 1000000
    # initialize the simulation log
    # each line of the log, corresponding to one event, contains the following fields
    sim_log = {
        "structure":{},
        "behavior":{
            "initial_marking": {},
            "firings": [],
            "final_marking": {}
        },
        "livelock":False
    }
    # log the ASN structure
    for t in pn_sn_list[0].transition():
        sim_log["structure"][t.name] = {}
        for pl,var in t.input():
            if not var.name in sim_log["structure"][t.name]:
                sim_log["structure"][t.name][var.name] = {"pre":[],"post":[]}
            sim_log["structure"][t.name][var.name]["pre"].append(pl.name)
        for pl,var in t.output():
            sim_log["structure"][t.name][var.name]["post"].append(pl.name)
    # log the ASN behavior
    # First, the initial marking
    m = pn_sn_list[0].get_marking()
    for pl_name in m:
        sim_log["behavior"]["initial_marking"][pl_name] = list(m(pl_name))
    sim_log["behavior"]["log_rows"] = []    
    # Then, do the simulation iterations by firining the enabled transitions
    iterations_count = 0
    e_t_count = len(pn_sn_list[0].transition())
    while e_t_count>0 and iterations_count<max_iterations_count and (len(pn_sn_list)>0):
        pn_sn_idx = get_pn_idx_for_sim_step(pn_sn_list)
        pn_sn = pn_sn_list[pn_sn_idx]
        # find enabling transition bindings
        e_t_bindings = {}
        for t in pn_sn.transition():
            b = t.modes()
            if len(b) > 0:
                e_t_bindings[t.name] = b
        e_t_count = len(e_t_bindings)
        if e_t_count > 0:
            # select one enabled transition using the uniform distribution
            sel_t_idx = int(rn.uniform(0,len(e_t_bindings)))
            sel_t_name = list(e_t_bindings)[sel_t_idx]
            sel_t = pn_sn.transition(sel_t_name)
            sel_t_bindings = e_t_bindings[sel_t_name]
            # select one of bindings of the selected enabled transition using the uniform distribution
            sel_b_idx = int(rn.uniform(0,len(sel_t_bindings)))
            sel_b = sel_t_bindings[sel_b_idx]
            # m_in,m_out = sel_t.flow(sel_b)
            sel_t.fire(sel_b)
            # fc_pc = int(100 * fc_count / total_case_count)
            # iter_pc = int(100*iterations_count/max_iterations_count)
            # print(f"{dt.datetime.now()}, iter:{iterations_count},{iter_pc}%, finished {fc_count} of {total_case_count} ({fc_pc}%), tr:{sel_t_name}, vars:{sel_b.dict()}")
            log_row = asm_logger(pn_pm4py,sel_t,sel_b)
            if not log_row is None:
                sim_log["behavior"]["log_rows"].append(log_row)
            sim_log["behavior"]["firings"].append((sel_t.name,sel_b.dict()))
            if is_final_marking_reached(pn_sn,pn_fm_pm4py):
                pn_sn_list.pop(pn_sn_idx)
        iterations_count += 1
    # And the final marking
    m = pn_sn.get_marking()
    for pl_name in m:
        sim_log["behavior"]["final_marking"][pl_name] = list(m(pl_name))
    sim_log['livelock'] = not iterations_count<max_iterations_count
    return sim_log