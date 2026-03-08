#! /bin/bash
#SBATCH -c 2
#SBATCH -N 1
#SBATCH --mem=12G
#SBATCH -p cpu  # Partition
#SBATCH -t 01:00:00  # Job time limit
#SBATCH -o sweep-%j.out  # %j = job ID
#SBATCH --array 0-100%50
#SBATCH --mail-type=END,FAIL

echo "SLURM_ATI: $SLURM_ARRAY_TASK_ID"
PARAM_STEP=0.01
PARAM_LOW=$(python3 -c "import os; print(float(os.getenv('PARAM_STEP')) * float(os.getenv('SLURM_ARRAY_TASK_ID')))")
PARAM_HIGH=$(python3 -c "import os; print(float(os.getenv('PARAM_STEP')) * (float(os.getenv('SLURM_ARRAY_TASK_ID'))+1))")
echo "Running DBSCAN with epsilon in [$PARAM_LOW, $PARAM_HIGH) and min_samples in [4, 10) on the training set..."
module load py-uv/0.4.27
uv run python -m steps.20_parameter_sweep --classifier dbscan_3param --dataset train --param eps=$PARAM_LOW:$PARAM_HIGH:$PARAM_STEP --param min_samples=4:10:1