This folder contains the Python code implementation of the Agent Mining discovery algorithm for the Agent Miner paper (ASM1).

How to run the Agent Miner algorithm:

1. Install the development and runtime environment (all software components are free for non-commercial use):
    a) Anaconda Python 3.8 distribution (https://www.anaconda.com/products/individual)
    b) Python Pandas (https://pandas.pydata.org/getting_started.html)
    c) Python NetworkX (https://networkx.org/)
    d) Python SNAKES (https://snakes.ibisc.univ-evry.fr/)
    e) Visual Studio Code IDE (https://code.visualstudio.com/)
    f) Visual Studio Code Python extention (https://code.visualstudio.com/docs/languages/python)
    g) Diagram editor yEd (https://www.yworks.com/products/yed)
    h) jBPT java library (https://github.com/jbpt/codebase/)
    i) PM4PY Process Mining library (https://pm4py.fit.fraunhofer.de/)
    j) ASM software code at GitHub (https://github.com/andreitour/at-phd-unimelb/tree/master/code/asm1)
    k) entropia java library for ntropy-based evaluation of Petri nets
    l) splitminer jaba library for discovery of Petri nets from event logs	 

2. To run the motivating example: 
    make asm1 the current folder
    download the motivating_example input logs from figshare into the asm1 folder
    run python script _pipeline_motivating_example.py

3. To run the BPIC evaluation:
    make asm1 the current folder
    download the bpic logs from the BPIC website into the corresponding sub-folders of asm1 ("bpic_2013","bpic_2020_TravelPermitData","bpic_2012","bpic_2017","bpic_2015_1","bpic_2014","bpic_2018","bpic_2019","bpic_2011")
    run python scripts:
      _pipeline_preprocess_bpic*.py (9 files)
      _pipeline_bpic_all_discover.py (to discover 4 types of models for all nine BPIC examples with 80% or 10% most frequent case variant coverage)
      _pipeline_bpic_all_eval.py (to evaluate the discovered models)
      _pipeline_bpic_all_viz.py (to generate PDF diagrams for the discovered models)