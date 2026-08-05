#!/bin/bash
#SBATCH --job-name=rnaduplex_10nt_real
#SBATCH --account=torch_pr_352_general

#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=1:00:00

#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -xeuo pipefail

echo "Hostname: $(hostname)"
echo "Starting at: $(date)"

cd /scratch/yl12316/interpretable-splicing-model-torch

# Real (unshuffled) MFE for the four 10-nt sub-regions of up_1_20 and ds_1_20.
#
# LEFT_FLANK_150[130:140] == LEFT_FLANK_130[110:120]  → up_11_20 (nt 11-20 from 3'SS)
# LEFT_FLANK_150[140:150] == LEFT_FLANK_130[120:130]  → up_1_10  (nt  1-10 from 3'SS)
# RIGHT_FLANK_100[0:10]                               → ds_1_10  (nt  1-10 from 5'SS)
# RIGHT_FLANK_100[10:20]                              → ds_11_20 (nt 11-20 from 5'SS)

singularity exec --nv \
  --overlay /scratch/yl12316/my_project/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif \
  /bin/bash -c "
    source /ext3/env.sh

    cd /scratch/yl12316/interpretable-splicing-model-torch

    python performance/rnaduplex_interact_flexible.py \
      --chunks \
        up_1_10:left:140:150 \
        up_11_20:left:130:140 \
        ds_1_10:right:0:10 \
        ds_11_20:right:10:20 \
      --out data/test_rnaduplex_finegrained_10nt.csv \
      --workers 16 \
      --force
  "

echo "Finished at: $(date)"
