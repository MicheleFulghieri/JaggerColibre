#!/bin/bash -l
#SBATCH --ntasks 1
#SBATCH --mem=100G
#SBATCH -J analysis
#SBATCH -o /cosma8/data/do019/dc-fulg1/outputs/first_analysis/simulation_%J.out
#SBATCH -e /cosma8/data/do019/dc-fulg1/outputs/first_analysis/simulation_%J.err
#SBATCH -p jagger
#SBATCH -A do019
#SBATCH -t 02:00:00       


module purge
module load cosma anaconda

# subsitute with eval(/cosma/local/anaconda3/202512/bin/conda $0 hook) or similar 
# according to logs
source $(conda info --base)/etc/profile.d/conda.sh
conda activate jagger

# Execute python script
python analysis/scripts/first_analysis.py
