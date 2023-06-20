#!/bin/bash

python3 -m venv pymarl/env

source pymarl/env/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

deactivate
