#!/bin/bash
#SBATCH --job-name=kl_reg_d90
#SBATCH --account=torch_pr_352_general

#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=72:00:00

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

    python -c 'import numpy, torch; print(\"numpy\", numpy.__version__); print(\"torch\", torch.__version__)'

    cd /scratch/yl12316/interpretable-splicing-model-torch

    python train.py \
      --train-npz data/train_data.npz \
      --test-npz data/test_data.npz \
      --input-length 90 \
      --checkpoint-dir checkpoints/kl_reg_d90 \
      --epochs 200 \
      --patience 20 \
      --batch-size 256 \
      --lr 1e-3 \
      --seed 42 \
      --no-batchnorm \
      --l1-lambda 1e-3 \
      --l2-smooth 1e-3
  "

echo "Finished at: $(date)"
