#!/bin/bash
#SBATCH --job-name=rnaduplex_multiseed
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
        ss3:left:107:130 \
        up_near:left:80:130 \
        up_mid:left:40:90 \
        up_far:left:0:50 \
        up_total:left:0:130 \
        upstream_exon:upstream_exon:0:116 \
        downstream_short:right:0:100 \
        downstream_long:downstream_long:0:250 \
        downstream_exon:downstream_exon:0:141 \
      --n-seeds 100 \
      --seed-start 42 \
      --out data/test_rnaduplex_redesigned_multiseed_100.csv \
      --workers 16 \
      --force
  "

echo "Finished at: $(date)"
