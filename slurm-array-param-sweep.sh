#! /bin/bash
#SBATCH -c 1
#SBATCH -N 1
#SBATCH --mem=12G
#SBATCH -p cpu  # Partition
#SBATCH -t 01:00:00  # Job time limit
#SBATCH -o slurm-%j.out  # %j = job ID

declare PARAM_LOW=0.01 * SLURM_ARRAY_TASK_ID
declare PARAM_HIGH=0.01 * (SLURM_ARRAY_TASK_ID + 1)
echo "Running DBSCAN with eps in [$PARAM_LOW, $PARAM_HIGH) and min_samples in [4, 10) on the training set..."
module load py-uv/0.4.27
uv run python -m steps.20_parameter_sweep --classifier dbscan --dataset train --param eps=$PARAM_LOW,$PARAM_HIGH:0.002 --param min_samples=4,10,1