#!/bin/bash

conda activate imp_marl_pymarl
#source imp_marl_venv/bin/activate

alg=$1
env=$2
python pymarl_train.py --config=${alg} --env-config=${env} with name=${alg}_${env} test_nepisode=-1

conda deactivate
#deactivate
