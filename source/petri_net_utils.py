import pm4py
import pandas as pd
import copy
from lxml import etree
from collections import defaultdict
import random
from numpy.random import choice

import sys
import os

from pm4py.algo.simulation.playout.petri_net import algorithm as pn_simulator_custom
from pm4py.visualization.petri_net import visualizer as pn_visualizer 
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.stochastic_petri import utils as stochastic_utils
from pm4py.algo.simulation.montecarlo.utils import replay
from pm4py.algo.simulation.playout.petri_net.variants.stochastic_playout import Parameters
from pm4py.objects import petri_net

def transform_df_to_log(log_df):
    log = copy.deepcopy(log_df)
    log.rename(columns={'start_timestamp': 'start_timestamp', 
                   'end_timestamp': 'time:timestamp',
                   'case_id': 'case:concept:name',
                   'activity_name': 'concept:name',}, inplace=True)
    log['case:concept:name'] = log['case:concept:name'].astype(str)
    log = log_converter.apply(log)
    return log

def discover_petri_net_inductive(log_df):
    log = transform_df_to_log(log_df)

    net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(
                                        log=log, 
                                        noise_threshold=0.0,
                                        timestamp_key='time:timestamp',)

    smap = get_stochastic_map(log, net, initial_marking, final_marking, activity_key='concept:name')

    return net, initial_marking, final_marking, smap


def write_petri_net(pn_pm4py,pn_im_pm4py,pn_fm_pm4py,pn_file_name_base):
    pm4py.write_pnml(pn_pm4py,pn_im_pm4py,pn_fm_pm4py,pn_file_name_base)


def get_stochastic_map(log, net, initial_marking, final_marking, activity_key='concept:name'):
    log['case:concept:name'] = log['case:concept:name'].astype(str)
    log['time:timestamp'] = pd.to_datetime(log['time:timestamp'], format='mixed')
    log = log_converter.apply(log)

    parameters = {}
    parameters_rep = copy.copy(parameters)
    parameters_rep[Parameters.ACTIVITY_KEY] = activity_key
    parameters_rep[Parameters.TIMESTAMP_KEY] = 'time:timestamp'
    smap = replay.get_map_from_log_and_net(log, net, initial_marking, final_marking,
                                               parameters=parameters_rep)
    
    return smap

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


def helper(et, smap):
    print(f"enabled transitions: {et}")
    wmap = {ct: smap[ct].get_weight() if ct in smap else 1.0 for ct in et}
    print(wmap)
    wmap_sv = sum(wmap.values())
    print(f"wmap_sv: {wmap_sv}")
    list_of_candidates = []
    probability_distribution = []
    for ct in wmap:
        list_of_candidates.append(ct)
        if wmap_sv == 0:
            probability_distribution.append(1.0/float(len(wmap)))
        else:
            probability_distribution.append(wmap[ct] / wmap_sv)
    print(f"probs: {probability_distribution}")


def get_enabled_transitions(net, marking):
    semantics=petri_net.semantics.ClassicSemantics()
    # get enabled transitions
    all_enabled_trans = semantics.enabled_transitions(net, marking)
    en_t_list = list(all_enabled_trans)
    print(f"enabled transitions: {en_t_list}")

    return en_t_list

def sample_next_role(net, marking, probabilities=None, current_role=None):
    en_t_list = get_enabled_transitions(net, marking)

    if probabilities == None:
        trans = random.choice(en_t_list)
    # elif type(probabilities) == dict:
    #     print(f"previous role: {current_role}")
    #     print(f"probabilities: {probabilities}")
    #     probs = probabilities[current_role].values()
    #     print(f"probs: {probs}")
    #     trans = random.choices(en_t_list, weights=probs, k=1)[0]
    else:
        # print(f"available roles: {en_t_list}")
        # print(f"probabilities: {probabilities}")
        helper(en_t_list, probabilities)
        trans = stochastic_utils.pick_transition(en_t_list, probabilities) # using smap
        # trans = pick_transition(en_t_list, smap=probabilities)

    return trans

def sample_next_activity(net, marking, final_marking, probabilities=None):
    semantics=petri_net.semantics.ClassicSemantics()
    en_t_list = get_enabled_transitions(net, marking)

    if probabilities == None:
        trans = random.choice(en_t_list)
    else:
        # fire most likely transition
        helper(en_t_list, probabilities)
        trans = stochastic_utils.pick_transition(en_t_list, probabilities)
        # trans = pick_transition(en_t_list, smap=probabilities)
        # print('most likely act')

    if semantics.execute(trans, net, marking) == final_marking:
        is_final_act = True
    else:
        is_final_act = False

    return trans, is_final_act

def update_marking_after_transition(trans, net, marking, final_marking=None):
    semantics=petri_net.semantics.ClassicSemantics()
    marking = semantics.execute(trans, net, marking)
    final_marking_not_reached = True
    print(f"uodated marking: {marking}")
    print(f"final_marking: {final_marking}")
    if final_marking != None:
        if marking == final_marking:
            final_marking_not_reached = False

    return marking, final_marking_not_reached

def fire_one_transition(net, en_t_list, smap, semantics, marking):
    # fire most likely transition
    trans = stochastic_utils.pick_transition(en_t_list, smap)
    # update marking in petri net
    marking = semantics.execute(trans, net, marking)

    return trans, marking




def get_custom_stochastic_map(log, net, initial_marking, final_marking, pn_path):
    aligned_log = align_log(log, net, initial_marking, final_marking)
    xor_petri = retrive_XOR_petrinet(pn_path)
    print(f"xor_petri: {xor_petri}")
    branching_probabilities = replay_traces_with_transitions(net, initial_marking, final_marking, aligned_log, xor_petri)
    # branching_probabilities = replay_traces_with_transitions(net, initial_marking, final_marking, log, xor_petri)
    print(f"branching_probabilities: {branching_probabilities}")

    return branching_probabilities


from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
def align_log(log_df, net, im, fm):
    log = transform_df_to_log(log_df)
    parameters = {alignments.Variants.VERSION_STATE_EQUATION_A_STAR.value.Parameters.PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE: True}
    aligned_log = []
    print(f"log: {log}")
    for trace in log:
        new_trace = []
        caseid = trace.attributes['concept:name']
        trace_log = pm4py.filter_trace_attribute_values(log, 'concept:name', [caseid], retain=True)
        new_trace = trace_log
        aligned_trace = alignments.apply_log(trace_log, net, im, fm, parameters=parameters)
        # print(f"aligned_trace: {aligned_trace}")
        # new_trace = extract_enriched_alignment(trace_log[0], aligned_trace)
        new_trace = extract_aligned_trace(aligned_trace, trace)
        aligned_log.append(new_trace)

    return aligned_log

def extract_aligned_trace(aligned_trace, original_trace):
    new_trace = []
    alignment = aligned_trace[0]['alignment']
    
    for move in alignment:
        if move[0][1] != '>>':  # If there's a move in the log
            new_trace.append(move[0][1])
    
    return new_trace


def retrive_XOR_petrinet(path_petrinet):
    """
    retrieve the XOR petri net from the petri net file - credits to Francesca Meneghello https://github.com/francescameneghello/RIMS/blob/RIMS_decision_points/decision_mining.py#L72
    """
    tree = etree.parse(path_petrinet)
    root = tree.getroot()
    net = root.find('net')
    page = net.find('page')

    xor_petri = defaultdict(list)

    for place in page.findall('place'):
        xor_petri[place.get('id')] = []

    for arc in page.findall('arc'):
        if arc.get('source') in xor_petri:
            xor_petri[arc.get('source')].append(arc.get('target'))

    remove = [key for key in xor_petri if len(xor_petri[key]) < 2]
    for r in remove:
        del xor_petri[r]

    return xor_petri


def replay_traces_with_transitions(petri_net, initial_marking, final_marking, traces, xor_petri):
    """
    Replays each trace through the Petri net, where each trace is a list of transitions.
    
    Args:
    - petri_net: The Petri net model.
    - initial_marking: The initial marking of the Petri net.
    - final_marking: The final marking of the Petri net.
    - traces: A list of traces, where each trace is a list of transitions (not activity names).
    
    Returns:
    - replay_results: A list of dictionaries for each trace, containing the visited places and transitions.
    """
    semantics=pm4py.objects.petri_net.semantics.ClassicSemantics()
    replay_results = []
    branching_frequencies = {place: {transition: 0 for transition in xor_petri[place]} for place in xor_petri}
    print(f"branching_frequencies: {branching_frequencies}")
    
    for trace in traces:
        print(f"new trace: {trace}")
        marking = initial_marking.copy() 
        for transition_name in trace:
            # Find the corresponding transition object
            transition = next((t for t in petri_net.transitions if t.name == transition_name), None)
            # check if current marking is a branching point in xor_petri
            # for mark in list(marking.keys()):
            #     if mark in xor_petri:
                    # Check if the transition exists in the Petri net
            if transition is not None:
                if semantics.is_enabled(transition, petri_net, marking):
                    # petri_net_trace.append(transition.name)
                    current_marking = list(marking.keys())
                    # print(f"current_marking: {current_marking}")
                    marking = semantics.execute(transition, petri_net, marking)
                    new_marking = list(marking.keys())
                    # print(f"marking after transition: {new_marking}")
                    # print(f"transition fired: {transition}")

                    for mark in current_marking:
                        if str(mark) in list(xor_petri.keys()):
                            if mark not in new_marking:
                                # this means that the transition has been fired and we are now at the next place
                                # track the transition and the place
                                # print(f"added {transition.name} to {mark}")
                                branching_frequencies[str(mark)][str(transition.name)] += 1
                                # print(f"branching_frequencies: {branching_frequencies}")
                                # petri_net_trace.append(mark)
                                # petri_net_trace.append(transition.name)
                            else:
                                # print(f"other transition fired: {mark}")
                                pass
                        else:
                            # print(f"not a branching point: {mark}")
                            pass
                    
                    # Add visited transition
                    # visited_transitions.append(transition.name)
                else:
                    raise Exception(f"Transition {transition.name} could not be fired due to marking {marking}")
            else:
                raise Exception(f"Transition {transition_name} not found in the Petri net.")

    # replay_results.append(petri_net_trace)
    print(f"branching_frequencies: {branching_frequencies}")
    # compute branching probabilities
    for place, transitions in branching_frequencies.items():
        total = sum(transitions.values())
        for k, v in transitions.items():
            branching_frequencies[place][k] = v / (total + 1e-6)
    
    return branching_frequencies