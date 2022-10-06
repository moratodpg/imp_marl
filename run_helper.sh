#!/bin/bash
algo=$1
id=$2
n_comp=10
discount_reward=100 # irrelevant for now
k_comp=100 # irrelevant for now
state_obs=True
state_d_rate=False
state_alphas=False
obs_d_rate=False
obs_multiple=False
obs_all_d_rate=False
obs_alphas=False
env_correlation=False
campaign_cost=False

./run_long.sh $algo $n_comp $discount_reward $k_comp $state_obs $state_d_rate $state_alphas $obs_d_rate $obs_multiple $obs_all_d_rate $obs_alphas $env_correlation $campaign_cost $id

