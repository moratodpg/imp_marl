import pytest
from environments.struct_env import Struct

def test_default_constructor():
    env = Struct()
    assert env.n_comp == 2
    assert env.discount_reward == 1
    assert env.k_comp == 1
    assert env.env_correlation == False
    assert env.campaign_cost == False

    obs = env.reset()
    assert obs.shape == (2, 2)