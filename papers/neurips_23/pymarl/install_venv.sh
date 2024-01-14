#!/bin/bash

python3 -m venv imp_marl_venv

source imp_marl_venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

deactivate
