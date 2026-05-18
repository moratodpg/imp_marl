import timeit

import numpy as np
import yaml

from agent_component_pf import ComponentPfAgent
from agent_interval import IntervalAgent
from agent_marginal_value import MarginalValueAgent
from agent_donothing import DoNothingAgent

AGENTS = {
    "interval": IntervalAgent,
    "marginal_value": MarginalValueAgent,
    "component_pf": ComponentPfAgent,
    "donothing": DoNothingAgent,
}


def make_env(env_cfg: dict):
    env_type = env_cfg["type"]
    params = {k: v for k, v in env_cfg.items() if k != "type"}
    if env_type == "struct":
        from imp_marl.environments.struct_env import Struct
        return Struct(params)
    if env_type == "owf":
        from imp_marl.environments.owf_env import Struct_owf
        return Struct_owf(params)
    if env_type == "zayas":
        from imp_marl.environments.struct_zayas import StructZayas
        return StructZayas(params)
    raise ValueError(f"Unknown env type: {env_type!r}")


def make_result_tag(env_cfg: dict, agent_type: str) -> str:
    t = env_cfg["type"]
    camp = "camp" if env_cfg.get("campaign_cost") else "ref"
    if t == "struct":
        corr = "c" if env_cfg.get("env_correlation") else "uc"
        return f"struct_{env_cfg['n_comp']}_{env_cfg.get('k_comp')}_{corr}_{camp}_{agent_type}"
    if t == "owf":
        return f"owf_{env_cfg['n_owt']}_{env_cfg['lev']}_{camp}_{agent_type}"
    if t == "zayas":
        return f"zayas_{env_cfg['n_comp']}_{camp}_{agent_type}"
    return f"{t}_{agent_type}"


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["search"].get("seed")
    if seed is not None:
        np.random.seed(seed)

    env = make_env(cfg["env"])
    agent_type = cfg["agent"]["type"]
    agent_kwargs = cfg["agent"].get("params", {}) or {}
    agent = AGENTS[agent_type](
        env,
        cfg["env"],
        result_tag=make_result_tag(cfg["env"], agent_type),
        **agent_kwargs,
    )

    start = timeit.default_timer()
    if cfg["search"]["enabled"]:
        opt = agent.search(cfg["search"]["eval_size"], cfg["search"]["params"])
        print("Optimal:", opt)
    else:
        eval_cfg = cfg["eval"]
        agent.eval(eval_cfg["eval_size"], **eval_cfg["params"])
    print(f"Time: {timeit.default_timer() - start:.1f}s")
