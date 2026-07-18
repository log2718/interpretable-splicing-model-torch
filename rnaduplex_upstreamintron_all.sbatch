#!/bin/bash
#SBATCH --account=torch_pr_352_general
#SBATCH --job-name=rnaduplex_up_total
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

cd /scratch/yl12316/interpretable-splicing-model-torch

singularity exec --nv --overlay /scratch/yl12316/my_project/overlay-15GB-500K.ext3:ro \
    /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif /bin/bash -c '
    source /ext3/env.sh
    cd /scratch/yl12316/interpretable-splicing-model-torch
    python performance/rnaduplex_interact_flexible.py \
        --chunks up_total:left:20:150 \
        --out data/test_rnaduplex_up_total.csv \
        --workers 16
'
