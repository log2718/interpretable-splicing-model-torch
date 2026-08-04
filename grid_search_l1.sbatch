#!/bin/bash
#SBATCH --job-name=grid_l1
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

# L1 values to sweep; L2 fixed at 1e-3 (position bias smoothness)
L1_VALUES=(1e-2 5e-2 1e-1)
L1=${L1_VALUES[$SLURM_ARRAY_TASK_ID]}
OUTDIR="checkpoints/grid_l1_${L1}"

echo "L1 lambda: ${L1}  checkpoint-dir: ${OUTDIR}"

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
      --checkpoint-dir ${OUTDIR} \
      --epochs 200 \
      --patience 20 \
      --batch-size 256 \
      --lr 1e-3 \
      --seed 42 \
      --no-batchnorm \
      --l1-lambda ${L1} \
      --l2-smooth 1e-3
  "

echo "Finished at: $(date)"
