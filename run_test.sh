#!/bin/bash
source env/bin/activate
python main_run_test.py --config=qvmix --env-config=struct with test_nepisode=300 env_args.components=4 env_args.state_config=obs use_tensorboard=True name=test_qvmix__2022-08-04_14-51-52 checkpoint_path=/Users/pascalleroy/Documents/struct_marl/results/models/qvmix__2022-08-09_16-29-47
deactivate

