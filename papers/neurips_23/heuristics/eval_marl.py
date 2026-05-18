"""Greedy evaluation of a trained QMIX agent on the Zayas environment.

Usage: python eval_marl.py
Configure model path, episodes, and observation settings in config.yaml [marl].
"""

import timeit
from types import SimpleNamespace

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import yaml

from imp_marl.environments.struct_zayas import StructZayas


# ── RNN agent (matches EPyMARL's RNNAgent) ─────────────────────────────────────

class RNNAgent(nn.Module):
    def __init__(self, input_shape: int, args):
        super().__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.rnn = (
            nn.GRUCell(args.hidden_dim, args.hidden_dim)
            if args.use_rnn
            else nn.Linear(args.hidden_dim, args.hidden_dim)
        )
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        h = self.rnn(x, h_in) if self.args.use_rnn else F.relu(self.rnn(x))
        return self.fc2(h), h


# ── Inference-only MAC ─────────────────────────────────────────────────────────

class EvalMAC:
    """Greedy multi-agent controller for evaluation. No EPyMARL dependency."""

    def __init__(self, input_shape: int, n_agents: int, args):
        self.n_agents = n_agents
        self.agent = RNNAgent(input_shape, args)
        self.hidden = None

    def load(self, checkpoint_dir: str):
        weights = th.load(f"{checkpoint_dir}/agent.th", map_location="cpu")
        self.agent.load_state_dict(weights)
        self.agent.eval()

    def init_hidden(self):
        h = self.agent.init_hidden()                              # (1, hidden_dim)
        self.hidden = h.expand(self.n_agents, -1).contiguous()   # (n_agents, hidden_dim)

    @th.no_grad()
    def select_actions(self, inputs: np.ndarray, avail_actions: np.ndarray) -> np.ndarray:
        """Greedy action selection.

        inputs:        (n_agents, input_shape)
        avail_actions: (n_agents, n_actions)  — 1 = available, 0 = masked
        returns:       (n_agents,) int array
        """
        q, self.hidden = self.agent(
            th.tensor(inputs, dtype=th.float32), self.hidden
        )
        q[th.tensor(avail_actions, dtype=th.float32) == 0] = -float("inf")
        return q.argmax(dim=-1).numpy()


# ── Observation builder ────────────────────────────────────────────────────────

def build_inputs(env: StructZayas, last_actions_oh: np.ndarray, marl_cfg: dict) -> np.ndarray:
    """Build (n_agents, input_shape) from current env state."""
    obs = np.stack([env.observations[a] for a in env.agent_list])  # (n, proba_size+1)
    parts = [obs]
    if marl_cfg.get("obs_d_rate", False):
        parts.append(env.d_rate[:, 0:1] / env.ep_length)          # (n, 1)
    if marl_cfg.get("obs_last_action", False):
        parts.append(last_actions_oh)                               # (n, n_actions)
    if marl_cfg.get("obs_agent_id", False):
        parts.append(np.eye(env.n_comp))                            # (n, n)
    return np.concatenate(parts, axis=1).astype(np.float32)


def get_input_shape(env: StructZayas, marl_cfg: dict) -> int:
    shape = env.proba_size + 1  # base obs: damage_proba + normalised time
    if marl_cfg.get("obs_d_rate", False):
        shape += 1
    if marl_cfg.get("obs_last_action", False):
        shape += marl_cfg["n_actions"]
    if marl_cfg.get("obs_agent_id", False):
        shape += env.n_comp
    return shape


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(env: StructZayas, mac: EvalMAC, marl_cfg: dict) -> float:
    env.reset()
    mac.init_hidden()

    n = env.n_comp
    n_actions = marl_cfg["n_actions"]
    avail_actions = np.ones((n, n_actions))        # all actions always available
    last_actions_oh = np.zeros((n, n_actions))
    total_reward = 0.0
    done = False

    while not done:
        inputs = build_inputs(env, last_actions_oh, marl_cfg)
        actions = mac.select_actions(inputs, avail_actions)

        action_dict = {env.agent_list[i]: int(actions[i]) for i in range(n)}
        _, rewards, done, _ = env.step(action_dict)
        total_reward += rewards[env.agent_list[0]]

        last_actions_oh = np.zeros((n, n_actions))
        last_actions_oh[np.arange(n), actions] = 1.0

    return total_reward


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    marl_cfg = cfg["marl"]
    env_cfg = {k: v for k, v in cfg["env"].items() if k != "type"}

    seed = marl_cfg.get("seed")
    if seed is not None:
        np.random.seed(seed)
        th.manual_seed(seed)

    env = StructZayas(env_cfg)

    args = SimpleNamespace(
        hidden_dim=marl_cfg["hidden_dim"],
        n_actions=marl_cfg["n_actions"],
        use_rnn=marl_cfg.get("use_rnn", True),
    )
    input_shape = get_input_shape(env, marl_cfg)
    mac = EvalMAC(input_shape, env.n_comp, args)
    mac.load(marl_cfg["model_dir"])
    print(f"Model:    {marl_cfg['model_dir']}  (input_shape={input_shape})")

    eval_episodes = marl_cfg["eval_episodes"]
    start = timeit.default_timer()
    returns = [run_episode(env, mac, marl_cfg) for _ in range(eval_episodes)]

    print(f"Episodes: {eval_episodes}")
    print(f"Return:   {np.mean(returns):.4f} ± {np.std(returns):.4f}")
    print(f"Time:     {timeit.default_timer() - start:.1f}s")
