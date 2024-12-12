import os
import datetime
import asm_pn as pn
print(datetime.datetime.now(),"=== START of import_example ===")

print("current folder:",os.getcwd())
out_dir = os.path.join(os.getcwd(),"agent_flowers_example_artem")
in_file_name = os.path.join(out_dir,"_source_artem_pn.pnml")
out_file_name_base = os.path.join(out_dir,"artem_pn")

from pm4py.objects.petri_net.importer import importer as pnml_importer
net, initial_marking, final_marking = pnml_importer.apply(in_file_name)
pn.viz_pn(net, initial_marking, final_marking,out_file_name_base)

print(datetime.datetime.now(),"=== END of import_example ===")  