# ──────────────────────────────────────────────────────────────────────────────
# run_vienna_flank_40_30.sh
#
# Fine-tune the PNAS model for the 40+70+30 = 140 nt flank experiment.
# Assumes train/test NPZ files are already prepared in data/.
#
# Usage:
#   ./run_vienna_flank_40_30.sh [epochs] [learning_rate]
#
# Examples:
#   ./run_vienna_flank_40_30.sh
#   ./run_vienna_flank_40_30.sh 50
#   ./run_vienna_flank_40_30.sh 50 5e-5
# ──────────────────────────────────────────────────────────────────────────────
set -e

EPOCHS="${1:-30}"
LR="${2:-1e-4}"
TAG="flank_40_30"

echo "=============================================="
echo " Fine-tuning PNAS model  (${TAG})"
echo "  Input length : 140 nt  (40 + 70 + 30)"
echo "  Checkpoint   : model_weights.pt"
echo "  Epochs       : ${EPOCHS}  LR: ${LR}"
echo "=============================================="

python train.py \
    --train-npz "data/train_${TAG}.npz" \
    --test-npz  "data/test_${TAG}.npz" \
    --checkpoint model_weights.pt \
    --input-length 140 \
    --no-batchnorm \
    --lr "${LR}" \
    --epochs "${EPOCHS}" \
    --checkpoint-dir "checkpoints/${TAG}"

echo ""
echo "=============================================="
echo "✓ Experiment complete (${TAG})"
echo "  Checkpoints: checkpoints/${TAG}/"
echo "=============================================="
