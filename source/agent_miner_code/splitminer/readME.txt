SM - Split Miner, version 2.0 - 3 June 2017

OVERVIEW:

SM is a standalone tool for fast discovery of simple and accurate process models in BPMN from an event log. 
The event log can be in MXML or XES format. The discovered model, in BPMN 2.0, can be opened and visualized using different tools, e.g. Apromore.



CONTENTS OF THIS DISTRIBUTION:

This distribution contains the following files and folders:
* splitminer.jar - Java console application (see usage below) : outputs BPMN 2.0 complete elements (outputs may contain OR gateways depending on the input settings)
* /lib - library folder (keep always in the same folder of the splitminer.jar file)
* /logs - sample of real life logs from https://data.4tu.nl/repository/collection:event_logs_real
* /outputs - outputs of the sample logs
* LICENSE.txt - licensing terms for SM

USAGE:

The tool requires the following mandatory inputs:
- parallelism threshold (epsilon): double in [0,1] 
- percentile for frequency threshold (eta): double in [0,1]
- boolean flag to remove OR joins: boolean {false|true}
- the path of the input log (extension required: .xes | .xes.gz | .mxml)
- the name of the output file (no extension required)

For example:

WINDOWS:
	java -cp splitminer.jar;lib\* au.edu.unimelb.services.ServiceProvider SMD 0.1 0.4 false logs/SEPSIS.xes.gz outputs/sepsis

MACOS/Unix:
	java -cp splitminer.jar:./lib/* au.edu.unimelb.services.ServiceProvider SMD 0.1 0.4 false logs/SEPSIS.xes.gz outputs/sepsis


PETRINET OUTPUT VARIANT:

	simply replace the code SMD with SMPN in the java command, e.g.:
	java -cp splitminer.jar:./lib/* au.edu.unimelb.services.ServiceProvider SMPN 0.1 0.4 false logs/SEPSIS.xes.gz outputs/sepsis

REQUIREMENTS:

 Java 8 or above.
 
FURTHER INFORMATION:

Please contact Adriano Augusto > email: a [dot] augusto [at] unimelb [dot] edu [dot] au 
