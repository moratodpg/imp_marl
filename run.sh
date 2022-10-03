#!/bin/bash
set -x

#
# Warning: $3 and $4 are for discount_reward and k_comp but currently not used.
source env/bin/activate
python main.py --config=$1 --env-config=struct with env_args.n_comp=$2 env_args.state_obs=$5 env_args.state_d_rate=$6 env_args.state_alphas=$7 env_args.obs_d_rate=$8 env_args.obs_multiple=$9 env_args.obs_all_d_rate=${10} env_args.obs_alphas=${11} env_args.env_correlation=${12} env_args.campaign_cost=${13} use_tensorboard=True name=new_req_algo_$1_n_comp_$2_discount_reward_$3_k_comp_$4_state_obs_$5_state_d_rate_$6_state_alphas_$7_obs_d_rate_$8_obs_multiple_$9_obs_all_d_rate_${10}_obs_alphas_${11}_env_correlation_${12}_campaign_cost_${13}_id_${14}
deactivate
