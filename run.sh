#!/bin/bash
set -x
#source env/bin/activate
python main.py --config=$1 --env-config=struct with use_tensorboard=True
#deactivate
