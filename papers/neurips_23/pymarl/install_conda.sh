#!/bin/bash

conda create -n imp_marl_pymarl python=3.7
conda activate imp_marl_pymarl
pip install -r pymarl/requirements.txt
conda deactivate
