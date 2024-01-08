# Expert-based heuristic policies

To find the best expert-based heuristic policy, one must conduct a search over the possible inspection intervals and number of components to be inspected.

This is done via the [run_heuristics](run_heuristics.py) script.

This script takes as input via its first lines the parameters of the environments and the parameters of the heuristic search.

Execute the script `download_heuristic_logs.sh` to retrieve the logs of the experiments conducted in the paper.

**Reproduce the results:** you can either run again the policy search to identify the optimised heuristics or directly evaluate the stored policies.

The policy search can be executed by indicating `search = True` in the script  `run_heuristics.py`. 

To re-run the policy evaluation corresponding to the optimised heuristics, you can directly test the stored policies in heur_search folder. In this case, specify `search = False` in the script `run_heuristics.py`.
 
For example, to compute the return resulting from the uncorrelated 4-out-of-5 environment:

* Check the optimized heuristics: 'insp_interv': 10, 'insp_comp': 5
* The seed was set up as 0 by default
* Execute run_heuristics.py

