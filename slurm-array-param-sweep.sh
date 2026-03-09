#! /bin/bash
#SBATCH -c 2
#SBATCH -N 1
#SBATCH --mem=14G
#SBATCH -p cpu  # Partition
#SBATCH -t 02:00:00  # Job time limit
#SBATCH -o sweep-%j.out  # %j = job ID
#SBATCH --array 1-100
#SBATCH --mail-type=END,FAIL

echo "SLURM_ATI: $SLURM_ARRAY_TASK_ID"
PARAM_STEP='0.001'
export PARAM_PER_JOB='0.002'
PARAM_LOW=$(python3 -c "import os; print(float(os.getenv('PARAM_PER_JOB')) * float(os.getenv('SLURM_ARRAY_TASK_ID')))")
PARAM_HIGH=$(python3 -c "import os; print(float(os.getenv('PARAM_PER_JOB')) * (float(os.getenv('SLURM_ARRAY_TASK_ID'))+1))")
echo "Running HDBSCAN_3Param_Norm with epsilon in [$PARAM_LOW, $PARAM_HIGH) and min_samples in [5, 20) on the training set..."
module load py-uv/0.4.27
uv run python -m steps.20_parameter_sweep --classifier hdbscan_3param_norm --dataset all --param cluster_selection_epsilon=$PARAM_LOW:$PARAM_HIGH:$PARAM_STEP --param min_samples=5:20:5