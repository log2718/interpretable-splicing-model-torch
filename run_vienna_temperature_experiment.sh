# ──────────────────────────────────────────────────────────────────────────────
# run_vienna_temperature_experiment.sh
#
# End-to-end pipeline for a ViennaRNA temperature experiment:
#   1. Prepare train NPZ with the given RNAfold temperature
#   2. Prepare test  NPZ with the given RNAfold temperature
#   3. Fine-tune the PNAS model from the pretrained checkpoint
#
# Usage:
#   ./run_vienna_temperature_experiment.sh <temperature> [epochs] [learning_rate]
#
# Examples:
#   ./run_vienna_temperature_experiment.sh 60
#   ./run_vienna_temperature_experiment.sh 60 50
#   ./run_vienna_temperature_experiment.sh 60 50 5e-5
# ──────────────────────────────────────────────────────────────────────────────
set -e  # Exit immediately on any command failure

# ── Argument handling ─────────────────────────────────────────────────────────
if [ -z "$1" ]; then
    echo "ERROR: RNAfold temperature is required."
    echo "Usage: $0 <temperature> [epochs] [learning_rate]"
    echo ""
    echo "  temperature    RNAfold temperature in Celsius (required)"
    echo "  epochs         Number of training epochs      (default: 30)"
    echo "  learning_rate  Learning rate for fine-tuning   (default: 1e-4)"
    exit 1
fi

TEMP="$1"
EPOCHS="${2:-30}"
LR="${3:-1e-4}"

echo "=============================================="
echo " ViennaRNA Temperature Experiment"
echo "=============================================="
echo "  Temperature : ${TEMP}°C"
echo "  Epochs      : ${EPOCHS}"
echo "  LR          : ${LR}"
echo "=============================================="
echo ""

# ── Step 1: Prepare train NPZ ────────────────────────────────────────────────
echo "──────────────────────────────────────────────"
echo "[1/3] Preparing train dataset (vienna${TEMP})..."
echo "──────────────────────────────────────────────"

python prepare_dataset.py \
    --input-csv data/train_data.csv \
    --output-path "data/train_vienna${TEMP}.npz" \
    --temperature "${TEMP}" \
    --output-csv "data/train_vienna${TEMP}_annotated.csv"

echo ""
echo "✓ Train NPZ saved to data/train_vienna${TEMP}.npz"
echo "✓ Train CSV saved to data/train_vienna${TEMP}_annotated.csv"
echo ""

# ── Step 2: Prepare test NPZ ─────────────────────────────────────────────────
echo "──────────────────────────────────────────────"
echo "[2/3] Preparing test dataset (vienna${TEMP})..."
echo "──────────────────────────────────────────────"

python prepare_dataset.py \
    --input-csv data/test_data.csv \
    --output-path "data/test_vienna${TEMP}.npz" \
    --temperature "${TEMP}" \
    --output-csv "data/test_vienna${TEMP}_annotated.csv"

echo ""
echo "✓ Test NPZ saved to data/test_vienna${TEMP}.npz"
echo "✓ Test CSV saved to data/test_vienna${TEMP}_annotated.csv"
echo ""

# ── Step 3: Fine-tune the model ──────────────────────────────────────────────
echo "──────────────────────────────────────────────"
echo "[3/3] Fine-tuning PNAS model (vienna${TEMP})..."
echo "      Checkpoint: model_weights.pt"
echo "      Epochs: ${EPOCHS}  LR: ${LR}"
echo "──────────────────────────────────────────────"

python train.py \
    --train-npz "data/train_vienna${TEMP}.npz" \
    --test-npz "data/test_vienna${TEMP}.npz" \
    --checkpoint model_weights.pt \
    --no-batchnorm \
    --lr "${LR}" \
    --epochs "${EPOCHS}" \
    --checkpoint-dir "checkpoints/vienna${TEMP}"

echo ""
echo "=============================================="
echo "✓ Experiment complete (vienna${TEMP})"
echo "  Checkpoints: checkpoints/vienna${TEMP}/"
echo "=============================================="
