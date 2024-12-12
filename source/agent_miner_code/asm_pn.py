import datetime
import networkx as nx
import pm4py
from pm4py.objects.conversion.dfg import converter as dfg_mining
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.visualization.process_tree import visualizer as pt_visualizer
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.objects.petri_net.utils import petri_utils
from bs4 import BeautifulSoup
from source.agent_miner_code import asm_data as da
import os
import platform
from pm4py.objects.petri_net.importer import importer as pnml_importer

#USED
def pn_pm4py2nx(pn_pm4py, initial_marking=None, final_marking=None):
    def set_node_attr_gml(pn_nx,node_id,attr_name="",attr_value=""):
        if not attr_name=="" and not attr_value=="":
            labels = [("n:"+node_id,"t"),(attr_name+":"+attr_value,"b")]
            pn_nx.nodes[node_id]['LabelGraphics'] = [{'text':l[0],'anchor':l[1]} for l in labels]
    pn_nx = nx.DiGraph()
    for pl in pn_pm4py.places:
        pn_nx.add_node(pl.name)
        # set_node_attr_gml(pn_nx,pl.name,attr_name="n",attr_value=pl.name)
        pl_props = {'w':150,'h':50}
        pl_props['type'] = "ellipse"
        pl_props['fill'] = "#7CB9E8" # blue
        if (not initial_marking is None) and (pl in initial_marking):
            pl_marking = initial_marking[pl]
            set_node_attr_gml(pn_nx,pl.name,attr_name="im",attr_value=str(pl_marking))
            pl_props['fill'] = "#FFFFFF" # white
        if (not final_marking is None) and (pl in final_marking):
            pl_marking = final_marking[pl]
            set_node_attr_gml(pn_nx,pl.name,attr_name="fm",attr_value=str(pl_marking))
            pl_props['fill'] = "#FFFFFF" # white
        pn_nx.nodes[pl.name]['graphics'] = pl_props
    for tr in pn_pm4py.transitions:
        pn_nx.add_node(tr.name)
        # set_node_attr_gml(pn_nx,tr.name,attr_name="n",attr_value=tr.name)
        tr_props = {'w':150,'h':50}
        tr_props['type'] = 'rectangle'
        tr_label = None if tr.label is None else tr.label.strip().strip('\n')
        if tr_label is None or len(tr_label)==0: # silent transition
            tr_props['fill'] = "#7CB9E8" # blue
        else: # labeled transition
            tr_props['fill'] = "#FFA500" # orange
            set_node_attr_gml(pn_nx,tr.name,attr_name="l",attr_value=tr_label)
        pn_nx.nodes[tr.name]['graphics'] = tr_props
        for in_arc in tr.in_arcs:
            in_pl = in_arc.source
            pn_nx.add_edge(in_pl.name,tr.name)
        for out_arc in tr.out_arcs:
            out_pl = out_arc.target
            pn_nx.add_edge(tr.name,out_pl.name)
    return pn_nx

#USED
def pn_nx2pm4py(pn_nx):
    def is_place(pn_nx,node_id):
        return (pn_nx.nodes[node_id]['graphics']['type']=='ellipse')
    def get_node_attr_gml(pn_nx,node_id,attr_name):
        attr_value = None
        if type(pn_nx.nodes[node_id]['LabelGraphics'])==list:
            for lg in pn_nx.nodes[node_id]['LabelGraphics']: 
                if 'text' in lg:
                    label = lg['text']
                    label_parts = label.split(":")
                    if len(label_parts)>1 and label_parts[0].strip()==attr_name:
                        attr_value = label_parts[1].strip()
        return attr_value
    pn_pm4py = PetriNet("loaded_petri_net")
    im_pm4py = Marking()
    fm_pm4py = Marking()
    name_node_pm4py_map = {}
    for n in pn_nx.nodes:
        if is_place(pn_nx,n):
            pl_name = get_node_attr_gml(pn_nx,n,"n")
            if pl_name is None: pl_name = n
            pl = petri_utils.add_place(pn_pm4py,name=pl_name)
            name_node_pm4py_map[n]=pl
            pl_initial_marking = get_node_attr_gml(pn_nx,n,'im')
            if not pl_initial_marking is None: im_pm4py[pl] = int(pl_initial_marking)
            pl_final_marking = get_node_attr_gml(pn_nx,n,'fm')
            if not pl_final_marking is None: fm_pm4py[pl] = int(pl_final_marking)
        else: # n is a transition
            tr_name = get_node_attr_gml(pn_nx,n,'n')
            if tr_name is None: tr_name = n
            tr_label = get_node_attr_gml(pn_nx,n,'l')
            tr = petri_utils.add_transition(pn_pm4py,name=tr_name,label=tr_label)
            name_node_pm4py_map[n]=tr
    for f,t in pn_nx.edges:
        from_node = name_node_pm4py_map[f]
        to_node = name_node_pm4py_map[t]
        petri_utils.add_arc_from_to(from_node,to_node, pn_pm4py)
    return (pn_pm4py,im_pm4py,fm_pm4py)

#USED
def discover_pn_imf_pm4py(log_pm4py,im_variant,noise_threshold=0.0):
    # possible values for im_variant:
    #    pm4py.algo.discovery.inductive.algorithm.Variants.IM
    #    pm4py.algo.discovery.inductive.algorithm.Variants.IMd
    #    pm4py.algo.discovery.inductive.algorithm.Variants.IMf
    #    pm4py.algo.discovery.inductive.algorithm.Variants.IM_CLEAN (used in pm4py.discover_petri_net_inductive)
    parameters = pm4py.utils.get_properties(log_pm4py)
    # print(parameters)
    # print(type(parameters))
    # print(im_variant.value)
    parameters['noise_threshold'] = noise_threshold
    # parameters[im_variant.value.Parameters.NOISE_THRESHOLD] = noise_threshold
    pt = pm4py.algo.discovery.inductive.algorithm.apply(log_pm4py, variant=im_variant, parameters=parameters)
    from pm4py.convert import convert_to_petri_net
    return convert_to_petri_net(pt)

def discover_pn_hm_pm4py(log_pm4py,dependency_threshold=0.0):
    # possible values for im_variant:
    #    pm4py.algo.discovery.inductive.algorithm.Variants.IM
    #    pm4py.algo.discovery.inductive.algorithm.Variants.IMd
    #    pm4py.algo.discovery.inductive.algorithm.Variants.IMf
    #    pm4py.algo.discovery.inductive.algorithm.Variants.IM_CLEAN (used in pm4py.discover_petri_net_inductive)
    net, im, fm = pm4py.discover_petri_net_heuristics(log_pm4py, dependency_threshold=dependency_threshold)
    return (net, im, fm)

def discover_pn_alpha_pm4py(log_pm4py):
    net, im, fm = pm4py.discover_petri_net_alpha(log_pm4py)
    return (net, im, fm)

def discover_pn_sm_pm4py(xes_log_file_name,frequency_threshold=0.1):
    os.system(f"mkdir __temp")
    pnml_file_name_base = os.path.join("__temp","_splitminer_net")
    split_miner_jar = os.path.join("splitminer","splitminer.jar")
    split_miner_lib = os.path.join("splitminer","lib","*")
    path_separator = ";" if platform.system()=="Windows" else ":"
    sm_cmd_line = f"java -cp {split_miner_jar}{path_separator}{split_miner_lib} au.edu.unimelb.services.ServiceProvider SMPN 0.1 {frequency_threshold} false {xes_log_file_name} {pnml_file_name_base}"
    os.system(sm_cmd_line)
    pn_pm4py,pn_im_pm4py,pn_fm_pm4py = pnml_importer.apply(pnml_file_name_base+".pnml")      
    start_pl = [pl for pl in pn_pm4py.places if len(pl.in_arcs)==0][0]
    pn_im_pm4py[start_pl] = 1                    
    return (pn_pm4py,pn_im_pm4py,pn_fm_pm4py)

def viz_pn(pn_pm4py,pn_im_pm4py,pn_fm_pm4py,pn_file_name_base):
    print(datetime.datetime.now(),f"Writing PN to NX GML file {pn_file_name_base}.gml ...") 
    pn_nx = pn_pm4py2nx(pn_pm4py,initial_marking=pn_im_pm4py,final_marking=pn_fm_pm4py)
    nx.write_gml(pn_nx,pn_file_name_base+".gml")
    print(datetime.datetime.now(),f"Writing PN GVIZ to file {pn_file_name_base}.pdf ...") 
    gviz = pn_visualizer.apply(pn_pm4py, pn_im_pm4py, pn_fm_pm4py, parameters={pn_visualizer.Variants.FREQUENCY.value.Parameters.FORMAT: "pdf"}, variant=pn_visualizer.Variants.FREQUENCY)
    pn_visualizer.save(gviz, pn_file_name_base+".pdf")

def write_pnml(pn_pm4py,pn_im_pm4py,pn_fm_pm4py,pn_file_name_base):
    pm4py.write_pnml(pn_pm4py,pn_im_pm4py,pn_fm_pm4py,pn_file_name_base+".pnml")
    fix_pnml_for_jbpt(pn_file_name_base)

def remove_agent_self_loops_in(in_pm4py):
    def is_agent_self_loop_silent_tr(tr):
        if tr.label is None: # check if tr is a silent transition
            if len(tr.in_arcs)==1 and len(tr.out_arcs)==1:
                in_pl = list(tr.in_arcs)[0].source
                out_pl = list(tr.out_arcs)[0].target
                if len(in_pl.in_arcs)==1 and len(out_pl.out_arcs)==1:
                    s_node = list(in_pl.in_arcs)[0].source
                    t_node = list(out_pl.out_arcs)[0].target
                    return (s_node.name == t_node.name) and (not s_node.label is None)
        return False
    # remove agent-self-loop silent transitions
    loop_transitions_to_remove = set()
    for tr in in_pm4py.transitions:
        if is_agent_self_loop_silent_tr(tr):
            loop_transitions_to_remove.add(tr)
    removed_tr_names = set()
    for tr_to_remove in loop_transitions_to_remove:
        petri_utils.remove_transition(in_pm4py,tr_to_remove)
        removed_tr_names.add(tr_to_remove.name)
    return (in_pm4py,removed_tr_names)

#USED
def ensure_agent_self_loops_in(in_pm4py):
    labeled_tr_list = [tr for tr in in_pm4py.transitions if not tr.label is None]
    for tr in labeled_tr_list:
        out_pl = list(tr.out_arcs)[0].target
        in_pl = list(tr.in_arcs)[0].source
        loop_tr = None
        for loop_arc1 in out_pl.out_arcs:
            if loop_arc1.target.label is None:
                if len(loop_arc1.target.out_arcs)==1 and list(loop_arc1.target.out_arcs)[0].target.name==in_pl.name:
                    loop_tr = loop_arc1.target
        if loop_tr is None:
            loop_tr = petri_utils.add_transition(in_pm4py,tr.label+"_loop")
            petri_utils.add_arc_from_to(out_pl,loop_tr,in_pm4py)
            petri_utils.add_arc_from_to(loop_tr,in_pl,in_pm4py)
    return in_pm4py

def integrate_mas_pn_pm4py_no_handovers(cluster_pn_map):
    i_mas_pn_pm4py = PetriNet("mas")
    source_pl = petri_utils.add_place(i_mas_pn_pm4py,name="source")
    im_pm4py = Marking()
    im_pm4py[source_pl] = 1
    source_tr = petri_utils.add_transition(i_mas_pn_pm4py,name="source_out",label=None)
    petri_utils.add_arc_from_to(source_pl,source_tr,i_mas_pn_pm4py)    
    sink_pl = petri_utils.add_place(i_mas_pn_pm4py,name="sink")
    fm_pm4py = Marking()
    fm_pm4py[sink_pl] = 1
    sink_tr = petri_utils.add_transition(i_mas_pn_pm4py,name="sink_in",label=None)
    petri_utils.add_arc_from_to(sink_tr,sink_pl,i_mas_pn_pm4py)
    for cluster_id in cluster_pn_map:
        cl_pn_pm4py, cl_im_pm4py, cl_fm_pm4py = cluster_pn_map[cluster_id]
        # add the cluster pn to the integrated mas pn
        added_i_mas_cl_nodes = {}
        cl_source_pl = None
        cl_sink_pl = None
        for pl in cl_pn_pm4py.places:
            i_pl = petri_utils.add_place(i_mas_pn_pm4py,f"{cluster_id}+{pl.name}")
            added_i_mas_cl_nodes[pl.name] = i_pl
            if pl in cl_im_pm4py:
                cl_source_pl = i_pl
                petri_utils.add_arc_from_to(source_tr,i_pl,i_mas_pn_pm4py)    
            if pl in cl_fm_pm4py:
                cl_sink_pl = i_pl
                petri_utils.add_arc_from_to(i_pl,sink_tr,i_mas_pn_pm4py)    
        for tr in cl_pn_pm4py.transitions:
            tr_label = tr.label # None if tr.label is None else da.get_activity_by_agent_label(tr.label,cluster_id)
            added_i_mas_cl_nodes[tr.name] = petri_utils.add_transition(i_mas_pn_pm4py,f"{cluster_id}+{tr.name}",tr_label)
        for arc in cl_pn_pm4py.arcs:
            from_node = added_i_mas_cl_nodes[arc.source.name]
            to_node = added_i_mas_cl_nodes[arc.target.name]
            petri_utils.add_arc_from_to(from_node,to_node,i_mas_pn_pm4py)
        # cl_loop_tr = petri_utils.add_transition(i_mas_pn_pm4py,name=cluster_id+"_loop_tr",label=None)
        # petri_utils.add_arc_from_to(cl_sink_pl,cl_loop_tr,i_mas_pn_pm4py)    
        # petri_utils.add_arc_from_to(cl_loop_tr,cl_source_pl,i_mas_pn_pm4py)    
    return (i_mas_pn_pm4py,im_pm4py,fm_pm4py)

#USED
def integrate_mas_pn_pm4py(hn_pm4py,hn_im_pm4py,hn_fm_pm4py,agent_pn_map=None):
    if (agent_pn_map is None) or len(agent_pn_map)==0:
        return (hn_pm4py,hn_im_pm4py,hn_fm_pm4py)
    else:        
        # add mas hn to the integrated mas pn 
        i_mas_pn_pm4py = petri_utils.merge(None,[hn_pm4py])
        print(f"agent pn map: {agent_pn_map}")
        # agent_pn_map = {('_others_' if k == ('_others_',) else k): v for k, v in agent_pn_map.items()}
        agent_pn_map = {key[0]: value for key, value in agent_pn_map.items()}
        # print(agent_pn_map.keys())
        # the conversation net (cn) has exactly one agent transition for a given agent (transition label == agent_id)
        print(f"hnpm4py: {hn_pm4py.transitions}")
        for agent_id_tr in hn_pm4py.transitions:
            if not agent_id_tr.label is None:
                agent_id = agent_id_tr.label
                i_cn_a_tr = petri_utils.get_transition_by_name(i_mas_pn_pm4py,agent_id_tr.name)
                print(f"agent id: {agent_id}")
                print(f"agent_pn_map: {agent_pn_map.keys()}")
                a_pn_pm4py, a_im_pm4py, a_fm_pm4py, a_dfg_obj = agent_pn_map[agent_id]
                # all agent nets ( as WF nets) are assumed to have one start (source) place
                a_source_pl = [pl for pl in a_pn_pm4py.places if pl in a_im_pm4py][0]
                # all cluster nets ( as WF nets) are assumed to have one end (sink) place
                a_sink_pl = [pl for pl in a_pn_pm4py.places if pl in a_fm_pm4py][0]

                # add the cluster pn to the integrated mas pn
                added_i_mas_a_nodes = {}
                for pl in a_pn_pm4py.places:
                    if (not pl.name==a_source_pl.name) and (not pl.name==a_sink_pl.name):
                        added_i_mas_a_nodes[pl.name] = petri_utils.add_place(i_mas_pn_pm4py,f"{agent_id}+{pl.name}")
                for tr in a_pn_pm4py.transitions:
                    tr_label = tr.label # None if tr.label is None else da.get_activity_by_agent_label(tr.label,cluster_id)
                    added_i_mas_a_nodes[tr.name] = petri_utils.add_transition(i_mas_pn_pm4py,f"{agent_id}+{tr.name}",tr_label)
                for a_arc in a_pn_pm4py.arcs:
                    if a_arc.source.name==a_source_pl.name:
                        from_to_list = [(i_cn_arc.source,added_i_mas_a_nodes[a_arc.target.name]) for i_cn_arc in i_cn_a_tr.in_arcs]
                    elif a_arc.source.name==a_sink_pl.name:
                        from_to_list = [(i_cn_arc.target,added_i_mas_a_nodes[a_arc.target.name]) for i_cn_arc in i_cn_a_tr.out_arcs]
                    elif a_arc.target.name==a_sink_pl.name:
                        from_to_list = [(added_i_mas_a_nodes[a_arc.source.name],i_cn_arc.target) for i_cn_arc in i_cn_a_tr.out_arcs]
                    elif a_arc.target.name==a_source_pl.name:
                        from_to_list = [(added_i_mas_a_nodes[a_arc.source.name],i_cn_arc.source) for i_cn_arc in i_cn_a_tr.in_arcs]
                    else :
                        from_to_list = [(added_i_mas_a_nodes[a_arc.source.name],added_i_mas_a_nodes[a_arc.target.name])]
                    for from_node,to_node in from_to_list:
                        petri_utils.add_arc_from_to(from_node,to_node,i_mas_pn_pm4py)
        
                # remove the hn cluster transition as it hs been replaced with the cluster pn
                petri_utils.remove_transition(i_mas_pn_pm4py,i_cn_a_tr)   
        return (i_mas_pn_pm4py,hn_im_pm4py,hn_fm_pm4py)

def integrate_mas_pn2_pm4py(agent_pn_map=None):
    if (agent_pn_map is None) or len(agent_pn_map)==0:
        return None
    else:        
        # add mas hn to the integrated mas pn 
        mas_pn_pm4py = PetriNet("loaded_petri_net")
        mas_source_pl = petri_utils.add_place(mas_pn_pm4py,"source")
        mas_sink_pl = petri_utils.add_place(mas_pn_pm4py,"sink")
        im_pm4py = Marking()
        im_pm4py[mas_source_pl]=1
        fm_pm4py = Marking()
        fm_pm4py[mas_sink_pl]=1
        msg_tr_map_map = {}
        # the conversation net (cn) has exactly one agent transition for a given agent (transition label == agent_id)
        for agent_id in agent_pn_map:
            if not agent_id=="_env":
                a_pn_pm4py, a_im_pm4py, a_fm_pm4py, a_dfg_obj = agent_pn_map[agent_id]
                a_node_map = {}
                for pl in a_pn_pm4py.places:
                    mas_pl = petri_utils.add_place(mas_pn_pm4py,f"{agent_id}+{pl.name}")
                    a_node_map[pl] = mas_pl
                for tr in a_pn_pm4py.transitions:
                    mas_tr = petri_utils.add_transition(mas_pn_pm4py,f"{agent_id}+{tr.name}",tr.label)   
                    a_node_map[tr] = mas_tr
                    if not mas_tr.label is None:
                        label_parts = mas_tr.label.split("|")
                        if len(label_parts)>1:
                            # this is a message transition (_msg_out or _msg_in)
                            msg_type = (label_parts[1],label_parts[2],label_parts[3],label_parts[4])
                            if not msg_type in msg_tr_map_map : 
                                msg_tr_map_map[msg_type] = {}
                            msg_tr_map_map[msg_type][label_parts[0]] = mas_tr
                for a_arc in a_pn_pm4py.arcs:
                    if (a_arc.source in a_node_map) and (a_arc.target in a_node_map):
                        from_node = a_node_map[a_arc.source]
                        to_node = a_node_map[a_arc.target]
                        petri_utils.add_arc_from_to(from_node,to_node,mas_pn_pm4py)
        for msg_type in msg_tr_map_map:
            msg_tr_map = msg_tr_map_map[msg_type]
            msg_out_tr = msg_tr_map["_msg_out"] if "_msg_out" in msg_tr_map else None
            msg_in_tr = msg_tr_map["_msg_in"] if "_msg_in" in msg_tr_map else None
            if msg_in_tr is None:
                # msg_out is sending mesage to the environment
                petri_utils.add_arc_from_to(msg_out_tr,mas_sink_pl,mas_pn_pm4py)
                msg_out_tr.label = None
            elif msg_out_tr is None:
                # msg_in is receiving msg from the environment
                petri_utils.add_arc_from_to(mas_source_pl,msg_in_tr,mas_pn_pm4py)
                msg_in_tr.label = None
            else:
                # merge out and in messages
                for in_tr_post_pl in petri_utils.post_set(msg_in_tr):
                    petri_utils.add_arc_from_to(msg_out_tr,in_tr_post_pl,mas_pn_pm4py)    
                petri_utils.remove_transition(mas_pn_pm4py,msg_in_tr)
                msg_out_tr.label = None
        return (mas_pn_pm4py,im_pm4py,fm_pm4py)

def integrate_mas_pn3_pm4py(ion_pm4py,ion_im_pm4py,ion_fm_pm4py,agent_pn_map=None):
    if (agent_pn_map is None) or len(agent_pn_map)==0:
        return (ion_pm4py,ion_im_pm4py,ion_fm_pm4py)
    else:        
        # add mas io-net to the integrated mas pn 
        mas_pn_pm4py = petri_utils.merge(None,[ion_pm4py])
        # all agent nets ( as WF nets) are assumed to have one start (source) place
        mas_source_pl = [pl for pl in mas_pn_pm4py.places if (not len(petri_utils.pre_set(pl))>0)][0]
        mas_im_pm4py = Marking()
        mas_im_pm4py[mas_source_pl] = 1
        # all cluster nets ( as WF nets) are assumed to have one end (sink) place
        mas_sink_pl = [pl for pl in mas_pn_pm4py.places if (not len(petri_utils.post_set(pl))>0)][0]
        mas_fm_pm4py = Marking()
        mas_fm_pm4py[mas_sink_pl] = 1
        a_i_tr_map = {}
        a_o_tr_map = {}
        for io_tr in mas_pn_pm4py.transitions:
            if not io_tr.label is None:
                l_parts = io_tr.label.split("|")
                a_id = l_parts[0]
                i_act = l_parts[1]
                o_act = l_parts[2]
                a_i_tr_map[(a_id,i_act)] = io_tr
                a_o_tr_map[(a_id,o_act)] = io_tr
        # the io-net has exactly one io transition for a given agent io path (transition label == (agent_id))
        for agent_id in agent_pn_map:
            if not agent_id=='_env':
                a_pn_pm4py, a_im_pm4py, a_fm_pm4py, a_dfg_obj = agent_pn_map[agent_id]
                # all agent nets ( as WF nets) are assumed to have one start (source) place
                a_source_pl = [pl for pl in a_pn_pm4py.places if pl in a_im_pm4py][0]
                # all cluster nets ( as WF nets) are assumed to have one end (sink) place
                a_sink_pl = [pl for pl in a_pn_pm4py.places if pl in a_fm_pm4py][0]
                a_node_map = {}
                mas_io_tr_set = set()
                for pl in a_pn_pm4py.places:
                    if (not pl.name==a_source_pl.name) and (not pl.name==a_sink_pl.name):
                        a_node_map[pl] = petri_utils.add_place(mas_pn_pm4py,f"{agent_id}+{pl.name}")
                for tr in a_pn_pm4py.transitions:
                    mas_tr = petri_utils.add_transition(mas_pn_pm4py,f"{agent_id}+{tr.name}",tr.label)
                    a_node_map[tr] = mas_tr
                    if not mas_tr.label is None:
                        label_parts = tr.label.split("|")
                        if len(label_parts) > 1:
                            o_a_id = label_parts[1]
                            o_act = label_parts[2]
                            i_a_id = label_parts[3]
                            i_act = label_parts[4]
                            if agent_id==o_a_id:
                                # this is an agent output activity
                                mas_io_tr = a_o_tr_map[(agent_id,o_act)]
                                for o_pl in petri_utils.post_set(mas_io_tr):
                                    petri_utils.add_arc_from_to(mas_tr,o_pl,mas_pn_pm4py)
                                mas_io_tr_set.add(mas_io_tr)
                            if agent_id==i_a_id:
                                # this is an agent input activity
                                mas_io_tr = a_i_tr_map[(agent_id,i_act)]
                                for i_pl in petri_utils.pre_set(mas_io_tr):
                                    petri_utils.add_arc_from_to(i_pl,mas_tr,mas_pn_pm4py)
                                mas_io_tr_set.add(mas_io_tr)
                for a_arc in a_pn_pm4py.arcs:
                    if (a_arc.source in a_node_map) and (a_arc.target in a_node_map):
                        from_node = a_node_map[a_arc.source]
                        to_node = a_node_map[a_arc.target]
                        petri_utils.add_arc_from_to(from_node,to_node,mas_pn_pm4py)
                for mas_io_tr in mas_io_tr_set:
                    petri_utils.remove_transition(mas_pn_pm4py,mas_io_tr)   
        return (mas_pn_pm4py,mas_im_pm4py,mas_fm_pm4py)

#USED
def fix_pnml_for_jbpt(pnml_file_name_base):
    in_file_name = pnml_file_name_base+'.pnml'
    fixed_d = None
    with open(in_file_name, 'r') as file_to_read:
        xml = file_to_read.read()
        bs_data = BeautifulSoup(xml, "xml")
        for el in bs_data.find_all('toolspecific'):
            if el.attrs['activity']=="$invisible$":
                silent_tr_el = el.parent
                text_el = silent_tr_el.findChild('text')
                text_el.clear()
        fixed_xml = bs_data
    out_file_name = pnml_file_name_base+'_jbpt.pnml'
    with open(out_file_name, 'w') as file_to_write:
        print(fixed_xml,file=file_to_write)

#USED
def discover_pn_from_dfg(agent_dfg_obj):
    pn_pm4py, im_pm4py, fm_pm4py = dfg_mining.apply(
        agent_dfg_obj['control_flows'],
        parameters = {'start_activities':agent_dfg_obj['input_activity_types'],'end_activities':agent_dfg_obj['output_activity_types']},
        variant=pm4py.objects.conversion.dfg.converter.Variants.VERSION_TO_PETRI_NET_INVISIBLES_NO_DUPLICATES)    
        # variant=pm4py.objects.conversion.dfg.converter.Variants.VERSION_TO_PETRI_NET_ACTIVITY_DEFINES_PLACE)    
    return (pn_pm4py, im_pm4py, fm_pm4py)

#USED
def discover_flower_pn(model_id,activity_type_multi_list):
    activity_types = [label for label in  activity_type_multi_list]
    pn_pm4py = PetriNet(model_id+"_pn")
    source_pl = petri_utils.add_place(pn_pm4py,name="source")
    source_tr_label = activity_types[0]
    source_tr = petri_utils.add_transition(pn_pm4py,name="source_out",label=source_tr_label)
    sink_pl = petri_utils.add_place(pn_pm4py,name="sink")
    sink_tr_label = activity_types[-1]
    sink_tr = petri_utils.add_transition(pn_pm4py,name="sink_in",label=sink_tr_label)
    mid_pl = petri_utils.add_place(pn_pm4py,name=str(model_id))
    petri_utils.add_arc_from_to(source_pl,source_tr,pn_pm4py)
    petri_utils.add_arc_from_to(source_tr,mid_pl,pn_pm4py)
    petri_utils.add_arc_from_to(mid_pl,sink_tr,pn_pm4py)
    petri_utils.add_arc_from_to(sink_tr,sink_pl,pn_pm4py)
    im_pm4py = Marking()
    im_pm4py[source_pl] = 1
    fm_pm4py = Marking()
    fm_pm4py[sink_pl] = 1
    for activity_type in activity_type_multi_list:
        activity_tr = petri_utils.add_transition(pn_pm4py,name=activity_type,label=activity_type)
        petri_utils.add_arc_from_to(mid_pl,activity_tr,pn_pm4py)
        petri_utils.add_arc_from_to(activity_tr,mid_pl,pn_pm4py)
    return (pn_pm4py, im_pm4py, fm_pm4py,None)

#USED
def apply_murata_reduction_rules(pn_pm4py, im_pm4py, fm_pm4py):
    def try_fsp_rule(pn_pm4py, im_pm4py, fm_pm4py):
        # applying Murata reduction rule (a) fusion of series places (FSP ) to remove silent transitions (Murata1989 fig.22a)
        def is_tr_to_be_reduced(tr,pn_pm4py, im_pm4py, fm_pm4py):
            if tr.label is None: # tr has to be silent
                pl_pre_set = petri_utils.pre_set(tr)
                pl_post_set = petri_utils.post_set(tr)
                if len(pl_pre_set)==1 and len(pl_post_set)==1: # tr has to have one input and one output
                    in_pl = list(pl_pre_set)[0]
                    out_pl = list(pl_post_set)[0]
                    if len(petri_utils.post_set(in_pl))==1: # tr's input place must have only one output
                        if in_pl in im_pm4py:
                            if (not out_pl in fm_pm4py) and (not len(petri_utils.pre_set(out_pl))>1):
                                return (in_pl,tr,out_pl)
                        elif out_pl in fm_pm4py:
                            if (not in_pl in im_pm4py) and (not len(petri_utils.post_set(in_pl))>1):
                                return (in_pl,tr,out_pl)    
                        elif (not in_pl in im_pm4py) and (not out_pl in fm_pm4py):
                            return (in_pl,tr,out_pl)
            return None                        
        tr_removed = True
        while tr_removed:
            tr_removed = False
            in_tr_out = None
            for tr1 in pn_pm4py.transitions:
                in_tr_out = is_tr_to_be_reduced(tr1,pn_pm4py, im_pm4py, fm_pm4py)
                if not in_tr_out is None:
                    break
            if  not in_tr_out is None:
                in_pl, tr, out_pl = in_tr_out
                if (not in_pl in im_pm4py):
                    # remove tr
                    petri_utils.remove_transition(pn_pm4py,tr)
                    # merge in_pl input arcs into out_pl, remove in_pl
                    for in_pl_in_tr in petri_utils.pre_set(in_pl):
                        petri_utils.add_arc_from_to(in_pl_in_tr,out_pl,pn_pm4py)
                    petri_utils.remove_place(pn_pm4py,in_pl)
                else:
                    # remove tr
                    petri_utils.remove_transition(pn_pm4py,tr)
                    # merge out_pl input and output arcs into in_pl, remove out_pl
                    for out_pl_in_tr in petri_utils.pre_set(out_pl):
                        petri_utils.add_arc_from_to(out_pl_in_tr,in_pl,pn_pm4py)
                    for out_pl_out_tr in petri_utils.post_set(out_pl):
                        petri_utils.add_arc_from_to(in_pl,out_pl_out_tr,pn_pm4py)
                    petri_utils.remove_place(pn_pm4py,out_pl)
                tr_removed = True
        return (pn_pm4py, im_pm4py, fm_pm4py)

    def try_fst_rule(pn_pm4py, im_pm4py, fm_pm4py):
        # applying Murata reduction rule (b) fusion of series transitions (FST) to remove unmarked places (Murata1989 fig.22b)
        def is_pl_to_be_reduced(pl,pn_pm4py, im_pm4py, fm_pm4py):        
            if (not pl in im_pm4py) and (not pl in fm_pm4py): # do not reduce places with markings
                tr_pre_set = petri_utils.pre_set(pl)
                tr_post_set = petri_utils.post_set(pl)
                if len(tr_pre_set)==1 and len(tr_post_set)==1: # pl has to have one input and one output
                    in_tr = list(tr_pre_set)[0]
                    out_tr = list(tr_post_set)[0]
                    if len(petri_utils.pre_set(out_tr))==1: # pl's output target must have only one input
                        if (in_tr.label is None) or (out_tr.label is None):
                            # either input or output transition has to be silent
                            return (in_tr,pl,out_tr)
            return None                
        pl_removed = True
        while pl_removed:
            in_pl_out = None
            for pl1 in pn_pm4py.places:
                in_pl_out = is_pl_to_be_reduced(pl1,pn_pm4py, im_pm4py, fm_pm4py)
                if not in_pl_out is None:
                    break
            if  in_pl_out is None:
                pl_removed = False
            else:
                in_tr, pl, out_tr = in_pl_out
                petri_utils.remove_place(pn_pm4py,pl)
                if out_tr.label is None:
                    # apply rule
                    # merge out_tr outputs into in_tr, remove out_tr
                    for out_tr_out_pl in petri_utils.post_set(out_tr):
                        petri_utils.add_arc_from_to(in_tr,out_tr_out_pl,pn_pm4py)
                    petri_utils.remove_transition(pn_pm4py,out_tr)
                else: # in_tr is None
                    # merge in_tr inputs and outputs into out_pl, remove in_tr
                    for in_tr_in_pl in petri_utils.pre_set(in_tr):
                        petri_utils.add_arc_from_to(in_tr_in_pl,out_tr,pn_pm4py)
                    for in_tr_out_pl in petri_utils.post_set(in_tr):
                        petri_utils.add_arc_from_to(out_tr,in_tr_out_pl,pn_pm4py)
                    petri_utils.remove_transition(pn_pm4py,in_tr)
                pl_removed = True
        return (pn_pm4py, im_pm4py, fm_pm4py)
    
    pn_pm4py, im_pm4py, fm_pm4py = try_fsp_rule(pn_pm4py, im_pm4py, fm_pm4py)
    #pn_pm4py, im_pm4py, fm_pm4py = try_fst_rule(pn_pm4py, im_pm4py, fm_pm4py)
    return (pn_pm4py, im_pm4py, fm_pm4py)