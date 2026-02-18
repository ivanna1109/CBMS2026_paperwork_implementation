#!/bin/bash
# set the number of nodes and processes per node

#SBATCH --job-name=comp_train
# set the number of nodes and processes per node
#SBATCH --partition=cuda
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n20


# set max wallclock time
#SBATCH --time=24:00:00
#SBATCH --output=/home/ivanam/CBMS_bioWork/train_experiments/output/comp_%j.log
#SBATCH --error=/home/ivanam/CBMS_bioWork/train_experiments/output/comp_%j.err


source /etc/profile.d/modules.sh
module load python/miniconda3.10 
eval "$(conda shell.bash hook)"
conda activate cbms_biosnap

PYTHON_EXECUTABLE=$(which python)

${PYTHON_EXECUTABLE} -u /home/ivanam/CBMS_bioWork/train_experiments/train_and_full_comparison.py

echo "All files done."