#!/bin/bash
source env/bin/activate
alg=$1
env=$2
path="mypath"
ntest= 300
name=test_${alg}_${env}_aze123
python main_run_test.py --config=${alg} --env-config=${env} with test_nepisode=${ntest} checkpoint_path=${path} name=${name}
deactivate

