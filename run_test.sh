#!/bin/bash
source env/bin/activate

exp_dir="path"
exp_name=$1
alg=$(echo $exp_name | cut -d'_' -f1-3)
env=$(echo $exp_name | cut -d'_' -f4-6)
path=${exp_dir}${exp_name}
n_test=$2
n_env=8
name=test_${n_test}_${exp_name}
python main_run_test.py --config=${alg} --env-config=${env} with test_nepisode=${n_test} checkpoint_path=${path} runner=parallel batch_size_run=${n_env} name=${name}
deactivate

