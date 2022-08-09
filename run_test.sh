#!/bin/bash
set -x
source env/bin/activate
python main.py --config=qvmix --env-config=struct with env_args.components=4 env_args.state_config=obs use_tensorboard=True name=test_qvmix__2022-08-04_14-51-52 checkpoint_path=/path_to_models/qvmix__2022-08-09_16-29-47
deactivate

