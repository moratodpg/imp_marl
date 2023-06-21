#!/bin/bash

# WARNING: this script considers you are in the results_scripts folder.

# Create the logs directory, which is named logs
mkdir -p logs

# Download only the logs from zenodo.
wget https://zenodo.org/record/8032339/files/MARL_logs.zip

# Unzip the logs
unzip MARL_logs.zip -d logs/

# Go to the folder and unzip the zipped folders
cd logs/MARL_logs

unzip owf.zip
unzip struct_c.zip
unzip struct_uc.zip
