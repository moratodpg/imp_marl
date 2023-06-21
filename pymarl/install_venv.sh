#!/bin/bash

python3 -m venv pymarl/imp_marl_venv

source pymarl/imp_marl_venv/bin/activate

python -m pip install --upgrade pip
pip install -r pymarl/requirements.txt

deactivate
