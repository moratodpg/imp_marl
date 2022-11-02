#!/bin/bash
set -x
source env/bin/activate
alg=$1
env=$2
python main.py --config=${alg} --env-config=${env} with name=${alg}_${env}_$3
deactivate