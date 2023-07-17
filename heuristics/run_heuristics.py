from heuristics.heuristics_interval_owf import HeuristicsOwf
from heuristics.heuristics_intervals_struct import HeuristicsStruct
import timeit

if __name__ == '__main__':

    search = False
    env = "struct"  # "struct" or "owf"
    eval_size = 1500

    if env == "struct":
        n_comp = 5
        discount_reward = 0.95
        k_comp = 4
        env_correlation = False
        campaign_cost = False
        seed_test = 0
        heuristic = HeuristicsStruct(n_comp=n_comp,
                                     discount_reward=discount_reward,
                                     k_comp=k_comp,
                                     env_correlation=env_correlation,
                                     campaign_cost=campaign_cost,
                                     seed=seed_test)
    elif env == "owf":
        n_owt = 2
        discount_reward = 0.95
        lev = 3
        campaign_cost = False
        seed_test = 0
        heuristic = HeuristicsOwf(n_owt=n_owt,
                                  lev=lev,
                                  discount_reward=discount_reward,
                                  campaign_cost=False,
                                  seed=seed_test)
    else:
        heuristic = None

    if search:
        #### Search
        starting_time = timeit.default_timer()
        heuristic.search(eval_size)
        print("Time (s):", timeit.default_timer() - starting_time)

    else:
        #### Evaluation
        insp_int = 10
        insp_comp = 5
        heuristic.eval(eval_size, insp_int, insp_comp)
