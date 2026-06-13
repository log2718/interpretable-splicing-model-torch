#!/bin/bash
#SBATCH --job-name=vienna60_uncertainty
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
      --train-npz data/train_vienna60.npz \
      --test-npz data/test_vienna60.npz \
      --input-length 90 \
      --checkpoint checkpoints/vienna60/best_model_20260515_150124.pt \
      --checkpoint-dir checkpoints/vienna60_uncertainty_v2 \
      --uncertainty \
      --freeze-epochs 5 \
      --lr 1e-3 \
      --lr-phase2 1e-4 \
      --lambda 1.0 \
      --epochs 100 \
      --patience 10 \
      --batch-size 256 \
      --seed 42 \
      --no-batchnorm
  "

echo "Finished at: $(date)"
