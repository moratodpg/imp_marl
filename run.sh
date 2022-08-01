#!/bin/bash
set -x
#source env/bin/activate
python main.py --config=$1 --env-config=struct with env_args.components=$2 use_tensorboard=True name=$1_n_comp_$2
#deactivate
