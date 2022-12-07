#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=struct_marl            # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=logs/%j.log   # Log-file (important!)
#SBATCH --cpus-per-task=2           # Number of CPU cores to allocate
#SBATCH --mem=4G                   # Memory to allocate per allocated CPU core
#SBATCH --time=12:00:00                # Max execution time
#SBATCH --partition=batch
#SBATCH --array=1-2

# Run your Python script
./run.sh $1 $2 $SLURM_ARRAY_TASK_ID