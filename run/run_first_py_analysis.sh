#!/bin/bash -l
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH -J jagger
#SBATCH -o /cosma8/data/do019/dc-fulg1/outputs/first_analysis/logs/simulation_%J.out
#SBATCH -e /cosma8/data/do019/dc-fulg1/outputs/first_analysis/logs/simulation_%J.err
#SBATCH -p cosma-analyse
#SBATCH -A do019
#SBATCH -t 02:00:00    
#SBATCH --export=NONE                     


# Conda activation
source /cosma/local/anaconda3/202512/etc/profile.d/conda.sh
conda activate jagger

# For python libraries to use all the cpus asked to slurm
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Execute python script
python analysis/scripts/first_analysis.py