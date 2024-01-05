import numpy as np
from numpy.testing import assert_array_equal

from environments.struct_env import Struct

def test_default_constructor():
    env = Struct()
    assert env.n_comp == 2
    assert env.discount_reward == 1
    assert env.k_comp == 1
    assert env.env_correlation == False
    assert env.campaign_cost == False

    obs = env.reset()
    assert type(obs) is dict
    assert len(obs) == 2
    assert "agent_0" in obs
    assert "agent_1" in obs
    assert len(obs["agent_0"]) == 31
    assert len(obs["agent_1"]) == 31

    actions = {}
    actions["agent_0"] = 0
    actions["agent_1"] = 0
    next_obs, rewards, done, info = env.step(actions)
    assert type(rewards) is dict
    assert type(done) is bool
    assert type(info) is dict
    assert len(next_obs) == 2
    assert "agent_0" in next_obs
    assert "agent_1" in next_obs
    assert len(next_obs["agent_0"]) == 31
    assert len(next_obs["agent_1"]) == 31
    assert len(rewards) == 2
    assert "agent_0" in rewards
    assert "agent_1" in rewards
    assert done == False

def test_terminal_state():
    env = Struct()
    cpt = 0
    done = False
    while not done:
        _, _, done, _ = env.step({"agent_0": np.random.randint(0, 3), "agent_1": np.random.randint(0, 3)})
        cpt += 1
    assert cpt == 30

def test_repair_initial_distrib():
    env = Struct()
    init_distrib = env.initial_damage_proba
    obs = env.reset()

    # we remove the last element which is the time
    assert_array_equal(init_distrib[0],obs["agent_0"][:-1])
    assert_array_equal(init_distrib[1], obs["agent_1"][:-1])

    actions = {}
    actions["agent_0"] = 0
    actions["agent_1"] = 0
    _, _, done, _ = env.step(actions)

    # check repair gives the initial distribution
    actions["agent_0"] = 2
    actions["agent_1"] = 2
    _, _, done, _ = env.step(actions)

    assert_array_equal(init_distrib[0], obs["agent_0"][:-1])
    assert_array_equal(init_distrib[1], obs["agent_1"][:-1])

def test_only_nothing():
    np.random.seed(42)
    env = Struct()
    done = False
    total_reward = 0
    while not done:
        _, rewards, done, _ = env.step({"agent_0": 0, "agent_1": 0})
        total_reward += rewards["agent_0"]
    assert total_reward == -40.24126096000002

def test_only_inspect():
    np.random.seed(42)
    env = Struct()
    done = False
    total_reward = 0
    while not done:
        _, rewards, done, _ = env.step({"agent_0": 1, "agent_1": 1})
        total_reward += rewards["agent_0"]
    assert total_reward == -60.14257297286546

def test_only_repair():
    np.random.seed(42)
    env = Struct()
    done = False
    total_reward = 0
    while not done:
        _, rewards, done, _ = env.step({"agent_0": 2, "agent_1": 2})
        total_reward += rewards["agent_0"]
    assert total_reward == -1200.0