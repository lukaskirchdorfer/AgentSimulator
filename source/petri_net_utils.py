import pm4py
import copy
from numpy.random import choice
import os

from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.simulation.montecarlo.utils import replay
from pm4py.algo.simulation.playout.petri_net.variants.stochastic_playout import Parameters
from pm4py.objects import petri_net

def discover_petri_net_inductive(log_df, data_dir):
    log = copy.deepcopy(log_df)
    print(log)
    log.rename(columns={'start_timestamp': 'start_timestamp', 
                   'end_timestamp': 'time:timestamp',
                   'case_id': 'case:concept:name',
                   'activity_name': 'concept:name',
                   'resource': 'org:resource'}, inplace=True)
    log['case:concept:name'] = log['case:concept:name'].astype(str)
    # log = dataframe_utils.convert_timestamp_columns_in_df(log)
    log = log_converter.apply(log)

    net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(
                                        log=log, 
                                        noise_threshold=0.0,
                                        timestamp_key='start_timestamp',)

    smap = get_stochastic_map(log, net, initial_marking, final_marking, activity_key='concept:name')
    wmap = {ct: smap[ct].get_weight() for ct in smap.keys()}
    print(f"wmap: {wmap}")
    branching_probabilities = smap
    pn_path = os.path.join(data_dir,"petri_net.pnml")
    write_petri_net(net, initial_marking, final_marking, pn_path)

    return net, initial_marking, final_marking, branching_probabilities, pn_path


def write_petri_net(pn_pm4py,pn_im_pm4py,pn_fm_pm4py,pn_file_name_base):
    pm4py.write_pnml(pn_pm4py,pn_im_pm4py,pn_fm_pm4py,pn_file_name_base)


def pick_transition(et, smap):
    """
    Pick a transition in a set of transitions based on the weights specified by the stochastic map

    Parameters
    --------------
    et
        Enabled transitions
    smap
        Stochastic map

    Returns
    --------------
    trans
        Transition chosen according to the weights
    """ 
    # Try matching based on name and label
    for ct in et:
        matching_trans = next((t for t in smap.keys() 
                             if t.name == ct.name and t.label == ct.label), None)
        print(f"Transition {ct.name}/{ct.label} match found: {matching_trans is not None}")
    # Create wmap using property matching
    wmap = {}
    for ct in et:
        matching_trans = next((t for t in smap.keys() 
                             if t.name == ct.name and t.label == ct.label), None)
        if matching_trans:
            wmap[ct] = smap[matching_trans].get_weight()
        else:
            wmap[ct] = 1.0
    wmap_sv = sum(wmap.values())
    list_of_candidates = []
    probability_distribution = []
    for ct in wmap:
        list_of_candidates.append(ct)
        if wmap_sv == 0:
            probability_distribution.append(1.0/float(len(wmap)))
        else:
            probability_distribution.append(wmap[ct] / wmap_sv)
    ct = list(choice(et, 1, p=probability_distribution))[0]
    return ct



def get_stochastic_map(log, net, initial_marking, final_marking, activity_key='concept:name'):
    parameters = {}
    parameters_rep = copy.copy(parameters)
    parameters_rep[Parameters.ACTIVITY_KEY] = activity_key
    parameters_rep[Parameters.TIMESTAMP_KEY] = 'start_timestamp'
    smap = replay.get_map_from_log_and_net(log, net, initial_marking, final_marking,
                                               parameters=parameters_rep)
    
    return smap



def get_enabled_transitions(net, marking):
    """
    Get enabled transitions from the current marking
    Parameters:
        net (PetriNet): Petri net
        marking (Marking): Current marking
    Returns:
        list: List of enabled transitions
    """
    semantics=petri_net.semantics.ClassicSemantics()
    # get enabled transitions
    all_enabled_trans = semantics.enabled_transitions(net, marking)
    en_t_list = list(all_enabled_trans)
    print(f"enabled transitions: {en_t_list}")

    return en_t_list

def sample_next_activity(net, marking, smap, final_marking,  activities_performed, max_count):
    """
    Sample the next activity based on the stochastic map that indicates branching probabilities
    Parameters:
        net (PetriNet): Petri net
        marking (Marking): Current marking
        smap (dict): Stochastic map
        final_marking (Marking): Final marking
        activities_performed (list): List of activities performed
        max_count (dict): Maximum count of each activity
    Returns:
        trans: Next transition
        is_final_act: Boolean indicating if it's the final activity
    """
    semantics=petri_net.semantics.ClassicSemantics()
    # get enabled transitions
    en_t_list = get_enabled_transitions(net, marking)
    trans = pick_transition(en_t_list, smap)

    if semantics.execute(trans, net, marking) == final_marking:
        is_final_act = True
    else:
        is_final_act = False

    return trans, is_final_act

def update_marking_after_activity_executed(trans, net, marking):
    semantics=petri_net.semantics.ClassicSemantics()
    marking = semantics.execute(trans, net, marking)

    return marking