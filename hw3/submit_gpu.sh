#!/bin/sh
#SBATCH --time=6:00:00          # Maximum run time in hh:mm:ss
#SBATCH --mem=16000             # Maximum memory required (in megabytes)
#SBATCH --job-name=default_479  # Job name (to track progress)
#SBATCH --partition=cse479 # Partition on which to run job
#SBATCH --gres=gpu:1            # Don't change this, it requests a GPU

module load anaconda
conda activate tensorflow-env
# This line runs everything that is put after "sbatch submit_gpu.sh ..."
$@
