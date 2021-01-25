#!/bin/bash
#SBATCH --job-name=train_gat
#SBATCH --output=%x.o%j
#SBATCH --error=%x.o%j
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --mem=8gb
#SBATCH --cpus-per-task=8

# To clean and load modules defined at the compile and link phases
module purge

# Activate anaconda environment
source activate /gpfs/workdir/blampeyq/anaconda3/envs/stylo
export LD_LIBRARY_PATH=/gpfs/workdir/blampeyq/lib

# To compute in the submission directory
cd ${SLURM_SUBMIT_DIR}

# Execution
python -u /workdir/blampeyq/graph_attention_network/train_ppi.py --gpu 0 --epochs 20 --model GAT