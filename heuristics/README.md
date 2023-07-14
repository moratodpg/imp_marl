# Expert-based heuristic policies

To find the best expert-based heuristic policy, one must conduct a search over the possible inspection intervals and number of components to be inspected.

This is done via the [run_heuristics](run_heuristics.py) script.

This script takes as input via its first lines the parameters of the environments and the parameters of the heuristic search.

Execute the script `download_heuristic_logs.sh` to retrieve the logs of the experiments conducted in the paper.

**To reproduce** the results, please specify the seed and heuristics corresponding to the environment of interest. You can find those in the numpy files stored in heur_search folder.

For instance, to retrieve the test results of the environment uncorrelated 4-out-of-5:

* Check the optimized heuristics: 'insp_interv': 10, 'insp_comp': 5
* The seed was set up as 0 by default
* Execute run_heuristics.py

