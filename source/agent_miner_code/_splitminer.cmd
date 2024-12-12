set BASE_DIR=C:\Users\atour\Documents\_PhD\_Jupyter\asm_qm\asm2_2\bpic_2020_RequestForPayment\output\dfg_ff_1.0\cl_md_0.3
set LOG=%BASE_DIR%\std_cluster+activity_log.xes
set OUT=%BASE_DIR%\std_cluster+activity_sm

cd C:\Users\atour\Documents\_PhD\_splitminer
java -cp splitminer.jar;lib\* au.edu.unimelb.services.ServiceProvider SMPN 0.4 0.1 false %LOG% %OUT%