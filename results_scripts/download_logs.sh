#!/bin/bash

# Create the logs directory, which is name logs
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
