#!/bin/bash
set -x
#source env/bin/activate
python main.py --config=$1 --env-config=struct with env_args.components=$2 env_args.k_comp=$3 use_tensorboard=True name=$1_comp_$2_$3_run$4
#deactivate
