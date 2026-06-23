#!/bin/bash
#SBATCH --job-name=rnacofold_intrinsic
#SBATCH --account=torch_pr_352_general

#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2:00:00

#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -xeuo pipefail

echo "Hostname: $(hostname)"
echo "Starting at: $(date)"

cd /scratch/yl12316/interpretable-splicing-model-torch

singularity exec \
  --overlay /scratch/yl12316/my_project/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif \
  /bin/bash -c "
    source /ext3/env.sh

    which RNAcofold && RNAcofold --version

    cd /scratch/yl12316/interpretable-splicing-model-torch

    python performance/rnacofold_intrinsic.py \
      --csv data/test_annotated.csv \
      --out data/test_rnacofold_intrinsic.csv \
      --workers 16 \
      --temperature 37
  "

echo "Finished at: $(date)"
