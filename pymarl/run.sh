#!/bin/bash
set -x
conda activate marl_struct_gpu
alg=$1
env=$2
python main.py --config=${alg} --env-config=${env} with env_args.campaign_cost=$3 name=${alg}_${env}_cc_$3_$4 use_cuda=False test_nepisode=-1
deactivate
