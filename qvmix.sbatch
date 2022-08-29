#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=struct_marl            # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=logs/%j.log   # Log-file (important!)
#SBATCH --cpus-per-task=4           # Number of CPU cores to allocate
#SBATCH --mem=4G                   # Memory to allocate per allocated CPU core
#SBATCH --time=45:00                # Max execution time
#SBATCH --partition=all

# Activate your Anaconda environment
conda activate qVmix
# Run your Python script
cd /home/pleroy/struct_marl
./run.sh $1 $2 $3
