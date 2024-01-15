#!/bin/bash

conda activate imp_marl_pymarl
#source imp_marl_venv/bin/activate

exp_dir="results/models/" # directory where the trained models are stored
exp_name=$1
alg=$(echo $exp_name | cut -d'_' -f1-3)
env=$(echo $exp_name | cut -d'_' -f4-6)
if [[ $env == *"owf"* ]]; then
  env=$(echo $env | cut -d'_' -f1-2)
fi

path=${exp_dir}${exp_name}
n_test=$2 # number of test episodes
campaign_option=$3
n_env=2 # number of environments to run in parallel
name=test_${n_test}_${exp_name}
echo $alg
echo $env
python pymarl_test.py --config=${alg} --env-config=${env} with test_nepisode=${n_test} checkpoint_path=${path} runner=parallel batch_size_run=${n_env} use_cuda=True name=${name} env_args.campaign_cost=${campaign_option}

conda deactivate
#deactivate
