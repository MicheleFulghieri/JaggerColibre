#!/bin/bash -l
#SBATCH --ntasks 1
#SBATCH --mem=100G
#SBATCH -J jupyter
#SBATCH -o slurmjupyter%J.out
#SBATCH -e slurmjupyter%J.err
#SBATCH -p cosma-analyse
#SBATCH -A do019
#SBATCH -t 02:00:00  
#SBATCH --mail-type=END # notifications
#SBATCH --mail-user=m.fulghieri@campus.unimib.it

module purge
source /cosma/apps/do019/dc-fulg1/venvs/jagger/bin/activate

#Your venv will need jupyter installed (pip install jupyterlab)
export XDG_RUNTIME_DIR=""
#Run Jupyter
jupyter lab --no-browser --ip=`ifconfig | awk '/172.17/ {print $2}'`
