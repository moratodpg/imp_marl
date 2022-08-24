#!/bin/bash
set -x
#source env/bin/activate
python main.py --config=iql --env-config=struct_sarl with env_args.components=2 env_args.state_config=obs use_tensorboard=True name=$1_comp_$2_state_$3
#deactivate
