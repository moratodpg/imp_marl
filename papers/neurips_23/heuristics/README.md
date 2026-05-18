# Expert-based heuristic policies

Heuristic policies are searched and evaluated via [`run_heuristics.py`](run_heuristics.py), configured entirely through [`config.yaml`](config.yaml).

## Structure

| File | Description |
|---|---|
| `agent_base.py` | Abstract base class — shared `search()`, `eval()`, grid iteration, and result saving |
| `agent_interval.py` | `IntervalAgent` — inspect top-k components at fixed intervals; repair observed-damaged ones |
| `agent_marginal_value.py` | `MarginalValueAgent` — repair by marginal system-PF reduction; inspect by expected information value |
| `agent_component_pf.py` | `ComponentPfAgent` — threshold on per-component P(damaged) for both inspection and repair |
| `config.yaml` | Single configuration file: environment, agent type, search ranges, eval params |
| `run_heuristics.py` | Entry point |

## Agent strategies

**`interval`** — Compatible with `struct` and `owf` environments.
Every `insp_interval` steps, inspect the top `comp_inspection` components ranked by P(damaged). Repair any component whose last inspection revealed damage.

**`marginal_value`** — Compatible with `zayas` environment.
At each step, compute the marginal reduction in system failure probability that repairing each component would provide, and repair the highest-scoring ones above a threshold. Then inspect components with the highest expected information value.

**`component_pf`** — Compatible with all environments.
At each step, repair components where P(damaged) > `repair_comp_threshold`. If system PF > `insp_sys_threshold`, inspect the top `n_insp` components by P(damaged) that are not already being repaired.

## Usage

All configuration is set in `config.yaml`. Run from this directory:

```bash
python run_heuristics.py
```

### 1. Policy search

Set `search.enabled: true` in `config.yaml` and choose the environment, agent, and search parameter ranges:

```yaml
env:
  type: struct
  n_comp: 5
  k_comp: 4
  discount_reward: 0.95
  campaign_cost: false
  env_correlation: false

agent:
  type: interval

search:
  enabled: true
  eval_size: 100
  seed: 0
  params:
    insp_interval: [1, 30]
    comp_inspection: [1, 6]
```

Results are saved to `Results/<tag>_<timestamp>.npz`, which contains `ret_total` (all grid returns), `opt_heur` (best parameters and reward), and `config`.

### 2. Policy evaluation

Set `search.enabled: false` and specify the parameters to evaluate under `eval.params`:

```yaml
search:
  enabled: false

eval:
  eval_size: 1000
  params:
    insp_interval: 10
    comp_inspection: 5
```

### Switching agent types

Each agent uses different search parameters. The `config.yaml` file contains commented-out examples for all three agents. The relevant sections to swap are `agent.type` and `search.params`:

| Agent | Search params |
|---|---|
| `interval` | `insp_interval: [start, stop]`, `comp_inspection: [start, stop]` |
| `marginal_value` | `max_repairs: [start, stop]`, `comp_inspection: [start, stop]` |
| `component_pf` | `insp_sys_threshold: [v1, v2, ...]`, `repair_comp_threshold: [v1, v2, ...]`, `n_insp: [n1, n2, ...]` |

## Reproducing paper results

Execute `download_heuristic_logs.sh` to retrieve the experiment logs from the paper.

To verify a stored result, load the NPZ file and inspect `opt_heur`:

```python
import numpy as np

with np.load("../heur_search/results_struct_uc/heuristics_5_4ucref_2023_04_15_130930.npz", allow_pickle=True) as data:
    print(data["opt_heur"])
# e.g. {'opt_reward_mean': ..., 'insp_interval': 10, 'comp_inspection': 5}
```

Then set those values under `eval.params` in `config.yaml` (with `search.enabled: false`) and re-run.
