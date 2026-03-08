#! /bin/bash
#SBATCH -c 2
#SBATCH -N 1
#SBATCH --mem=12G
#SBATCH -p cpu  # Partition
#SBATCH -t 01:00:00  # Job time limit
#SBATCH -o sweep-%j.out  # %j = job ID
#SBATCH -array 0-100%50
#SBATCH --mail-type=END,FAIL

echo "SLURM_ATI: $SLURM_ARRAY_TASK_ID"
PARAM_LOW=$(python3 -c "import os; print(0.01 * float(os.getenv('SLURM_ARRAY_TASK_ID')))")
PARAM_HIGH=$(python3 -c "import os; print(0.01 * float(os.getenv('SLURM_ARRAY_TASK_ID'))+1)")
echo "Running HDBSCAN with epsilon in [$PARAM_LOW, $PARAM_HIGH) and min_samples in [4, 10) on the training set..."
module load py-uv/0.4.27
uv run python -m steps.20_parameter_sweep --classifier hdbscan --dataset train --param cluster_selection_epsilon=$PARAM_LOW:$PARAM_HIGH:0.01