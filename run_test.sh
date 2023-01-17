#!/bin/bash
source env/bin/activate

exp_dir="path"
exp_name=$1
alg=$(echo $exp_name | cut -d'_' -f1-3)
env=$(echo $exp_name | cut -d'_' -f4-6)
path=${exp_dir}${exp_name}
ntest=$2
name=test_${exp_name}
python main_run_test.py --config=${alg} --env-config=${env} with test_nepisode=${ntest} checkpoint_path=${path} name=${name}
deactivate

