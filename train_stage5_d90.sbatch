#!/bin/bash
#SBATCH --job-name=stage5_d90
#SBATCH --account=torch_pr_352_general
#SBATCH --array=0-2

#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=72:00:00

#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

set -xeuo pipefail

echo "Hostname: $(hostname)"
echo "Starting at: $(date)"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"

cd /scratch/yl12316/interpretable-splicing-model-torch

L1_VALUES=(2e-3 1e-2 2e-2)
L1=${L1_VALUES[$SLURM_ARRAY_TASK_ID]}

# Point to the existing stage 4 checkpoint
STAGE4_CKPT=$(ls checkpoints/staged4_d90_l1_${L1}/stage4/best_model_*.pt | tail -1)
OUTDIR="checkpoints/staged4_d90_l1_${L1}"

echo "L1 lambda: ${L1}  stage4 ckpt: ${STAGE4_CKPT}"

singularity exec --nv \
  --overlay /scratch/yl12316/my_project/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif \
  /bin/bash -c "
    source /ext3/env.sh

    python -c 'import numpy, torch; print(\"numpy\", numpy.__version__); print(\"torch\", torch.__version__)'

    cd /scratch/yl12316/interpretable-splicing-model-torch

    python train.py \
      --train-npz data/train_data.npz \
      --test-npz  data/test_data.npz \
      --input-length 90 \
      --checkpoint-dir ${OUTDIR} \
      --stage5-checkpoint ${STAGE4_CKPT} \
      --stage5-epochs 100 \
      --stage5-patience 10 \
      --stage5-lr 1e-4 \
      --batch-size 256 \
      --seed 42 \
      --no-batchnorm
  "

echo "Finished at: $(date)"
