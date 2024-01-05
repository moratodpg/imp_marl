import pytest
from imp_marl.imp_env.struct_env import Struct

def test_constructor():
    env = Struct(n_comp=3, discount_reward=0.5, k_comp=2, env_correlation=True, campaign_cost=True)
    assert env.n_comp == 3
    assert env.discount_reward == 0.5
    assert env.k_comp == 2
    assert env.env_correlation == True
    assert env.campaign_cost == True