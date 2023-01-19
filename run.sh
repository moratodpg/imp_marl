#!/bin/bash
set -x
source env/bin/activate
alg=iql_uc_100
env=struct_uc_100
python main.py --config=${alg} --env-config=${env} with name=${alg}_${env}_$3
deactivate