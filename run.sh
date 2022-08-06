#!/bin/bash
set -x
#source env/bin/activate
python main.py --config=$1 --env-config=struct with env_args.components=$2 env_args.state_config=$3 use_tensorboard=True name=$1_comp_$2_state_$3
#deactivate
