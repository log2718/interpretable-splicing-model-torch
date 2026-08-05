#!/bin/bash
#SBATCH --job-name=rnaduplex_10nt_shuf
#SBATCH --account=torch_pr_352_general

#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=20:00:00

#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -xeuo pipefail

echo "Hostname: $(hostname)"
echo "Starting at: $(date)"

cd /scratch/yl12316/interpretable-splicing-model-torch

# 100-seed mononucleotide-shuffle control for the four 10-nt sub-regions.
#
# Indices are relative to LEFT_FLANK_130 (used internally by rnaduplex_shuffle_multiseed.py):
#   up_1_10  : left:120:130  — nt  1-10 upstream of 3'SS
#   up_11_20 : left:110:120  — nt 11-20 upstream of 3'SS
#   ds_1_10  : right:0:10   — nt  1-10 downstream of 5'SS
#   ds_11_20 : right:10:20  — nt 11-20 downstream of 5'SS

singularity exec --nv \
  --overlay /scratch/yl12316/my_project/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif \
  /bin/bash -c "
    source /ext3/env.sh

    cd /scratch/yl12316/interpretable-splicing-model-torch

    python performance/rnaduplex_shuffle_multiseed.py \
      --chunks \
        up_1_10:left:120:130 \
        up_11_20:left:110:120 \
        ds_1_10:right:0:10 \
        ds_11_20:right:10:20 \
      --n-seeds 100 \
      --seed-start 42 \
      --out data/test_rnaduplex_finegrained_10nt_multiseed_100.csv \
      --workers 16 \
      --force
  "

echo "Finished at: $(date)"
