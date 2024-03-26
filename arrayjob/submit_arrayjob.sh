#!/bin/bash

# Specify resource requirements for the job
#SBATCH --time=00:15:00
#SBATCH --mem=200M
#SBATCH --output=arrayjob_%A_%a.out
#SBATCH --error=arrayjob_%A_%a.err
#SBATCH --array=0-3

# Activate the conda environment
source activate env/

# Each python process will be run in a separate task
python src/iris_knn.py --params-id $SLURM_ARRAY_TASK_ID