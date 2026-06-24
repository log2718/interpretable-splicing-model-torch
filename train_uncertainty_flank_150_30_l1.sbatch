#!/bin/bash
#SBATCH --job-name=flank150_unc_l1
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

    cd /scratch/yl12316/interpretable-splicing-model-torch

    python train.py \
      --train-npz data/train_flank_150_30.npz \
      --test-npz data/test_flank_150_30.npz \
      --input-length 250 \
      --checkpoint checkpoints/flank_150_30/best_model_20260530_042839.pt \
      --checkpoint-dir checkpoints/flank_150_30_uncertainty \
      --uncertainty \
      --freeze-epochs 5 \
      --lr 1e-3 \
      --lr-phase2 1e-4 \
      --lambda 1.0 \
      --l1-lambda 1e-3 \
      --epochs 100 \
      --patience 10 \
      --batch-size 256 \
      --seed 42 \
      --no-batchnorm
  "

echo "Finished at: $(date)"
