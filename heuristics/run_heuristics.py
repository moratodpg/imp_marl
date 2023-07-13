from heuristics.heuristics_interval_owf import HeuristicsOwf
from heuristics.heuristics_intervals_struct import HeuristicsStruct
import timeit

if __name__ == '__main__':

    search = True

    env = "owf"  # "struct" or "owf"

    eval_size = 2000

    discount_reward = 0.95

    if env == "struct":
        n_comp = 5
        k_comp = 4
        heuristic = HeuristicsStruct(n_comp,
                                     discount_reward,
                                     k_comp)
    elif env == "owf":
        n_owt = 2
        lev = 3
        heuristic = HeuristicsOwf(n_owt=n_owt,
                                  lev=lev,
                                  discount_reward=discount_reward)
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
