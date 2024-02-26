import numpy as np

from imp_marl.environments.owf_env import Struct_owf
from numpy.testing import assert_array_equal


def test_default_constructor():
    env = Struct_owf()
    assert env.n_owt == 2
    assert env.discount_reward == 1
    assert env.lev == 3
    assert env.campaign_cost is False

    obs = env.reset()
    assert type(obs) is dict
    assert len(obs) == 4
    assert "agent_0" in obs
    assert "agent_1" in obs
    assert len(obs["agent_0"]) == 61
    assert len(obs["agent_1"]) == 61
    assert len(obs["agent_2"]) == 61
    assert len(obs["agent_3"]) == 61

    actions = {}
    actions["agent_0"] = 0
    actions["agent_1"] = 0
    actions["agent_2"] = 0
    actions["agent_3"] = 0
    next_obs, rewards, done, info = env.step(actions)

    assert type(rewards) is dict
    assert type(done) is bool
    assert type(info) is dict
    assert len(next_obs) == 4
    assert "agent_0" in next_obs
    assert "agent_1" in next_obs
    assert len(next_obs["agent_0"]) == 61
    assert len(next_obs["agent_1"]) == 61
    assert len(next_obs["agent_2"]) == 61
    assert len(next_obs["agent_3"]) == 61
    assert len(rewards) == 4
    assert "agent_0" in rewards
    assert "agent_1" in rewards
    assert "agent_2" in rewards
    assert "agent_2" in rewards
    assert done is False


def test_terminal_state():
    env = Struct_owf()
    cpt = 0
    done = False
    while not done:
        _, rewards, done, _ = env.step(
            {
                "agent_0": np.random.randint(0, 3),
                "agent_1": np.random.randint(0, 3),
                "agent_2": np.random.randint(0, 3),
                "agent_3": np.random.randint(0, 3),
            }
        )
        assert rewards["agent_0"] == rewards["agent_1"]
        cpt += 1
    assert cpt == 20


def test_repair_initial_distrib():
    env = Struct_owf()
    init_distrib = env.initial_damage_proba
    obs = env.reset()

    # we remove the last element which is the time
    assert_array_equal(init_distrib[0][0], obs["agent_0"][:-1])
    assert_array_equal(init_distrib[0][1], obs["agent_1"][:-1])
    assert_array_equal(init_distrib[1][0], obs["agent_2"][:-1])
    assert_array_equal(init_distrib[1][1], obs["agent_3"][:-1])

    actions = {}
    actions["agent_0"] = 0
    actions["agent_1"] = 0
    actions["agent_2"] = 0
    actions["agent_3"] = 0
    _, _, done, _ = env.step(actions)

    # check repair gives the initial distribution
    actions["agent_0"] = 2
    actions["agent_1"] = 2
    actions["agent_2"] = 2
    actions["agent_3"] = 2
    _, _, done, _ = env.step(actions)

    assert_array_equal(init_distrib[0][0], obs["agent_0"][:-1])
    assert_array_equal(init_distrib[0][1], obs["agent_1"][:-1])
    assert_array_equal(init_distrib[1][0], obs["agent_2"][:-1])
    assert_array_equal(init_distrib[1][1], obs["agent_3"][:-1])
