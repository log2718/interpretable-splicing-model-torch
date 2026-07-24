#!/bin/bash
#SBATCH --job-name=rnaduplex_finegrained
#SBATCH --account=torch_pr_352_general

#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00

#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -xeuo pipefail

echo "Hostname: $(hostname)"
echo "Starting at: $(date)"

cd /scratch/yl12316/interpretable-splicing-model-torch

singularity exec --nv \
  --overlay /scratch/yl12316/my_project/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif \
  /bin/bash -c "
    source /ext3/env.sh

    cd /scratch/yl12316/interpretable-splicing-model-torch

    python performance/rnaduplex_shuffle_multiseed.py \
      --chunks \
        up_1_20:left:110:130 \
        up_21_40:left:90:110 \
        up_41_60:left:70:90 \
        ds_1_20:right:0:20 \
        ds_21_40:right:20:40 \
        ds_41_60:right:40:60 \
        ds_61_80:right:60:80 \
        ds_81_100:right:80:100 \
      --n-seeds 100 \
      --seed-start 42 \
      --out data/test_rnaduplex_finegrained_multiseed_100.csv \
      --workers 16 \
      --force
  "

echo "Finished at: $(date)"
