#!/bin/bash
#SBATCH --job-name=monitored_d90
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

SEEDS=(42 142 242)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

OUTDIR="checkpoints/monitored_seed${SEED}"
CSVLOG="logs/metrics_seed${SEED}.csv"

mkdir -p logs

echo "Seed: ${SEED}  checkpoint-dir: ${OUTDIR}  csv: ${CSVLOG}"

singularity exec --nv \
  --overlay /scratch/yl12316/my_project/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif \
  /bin/bash -c "
    source /ext3/env.sh

    python -c 'import numpy, torch; print(\"numpy\", numpy.__version__); print(\"torch\", torch.__version__)'

    cd /scratch/yl12316/interpretable-splicing-model-torch

    python train_monitored.py \
      --train-npz    data/train_data.npz \
      --test-npz     data/test_data.npz \
      --input-length 90 \
      --checkpoint-dir ${OUTDIR} \
      --csv-log        ${CSVLOG} \
      --staged \
      --stage1-epochs  50 \
      --stage2-epochs  50 \
      --epochs         100 \
      --stage4-epochs  100 \
      --stage5-epochs  100 \
      --patience       15 \
      --stage4-lr      1e-4 \
      --stage5-lr      1e-4 \
      --stage5-patience 10 \
      --batch-size     256 \
      --lr             1e-3 \
      --seed           ${SEED} \
      --no-batchnorm \
      --constant-var \
      --l1-lambda      2e-3 \
      --l2-smooth      1e-3
  "

echo "Finished at: $(date)"
