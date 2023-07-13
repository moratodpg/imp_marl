#!/bin/bash

# WARNING: this script considers you are in the results_scripts folder.

# Create the logs directory, which is named logs
mkdir -p heuristic_logs

# Download only the logs from zenodo.
wget https://zenodo.org/record/8032339/files/heur_logs.zip

# Unzip the logs
unzip heur_logs.zip -d heuristic_logs/
