"""
train_monitored.py — same 5-stage training as train.py but logs per-epoch
metrics to a CSV:
  seed, stage, epoch, train_loss, val_loss, test_kl, test_mse,
  sumdiff_w, sumdiff_b

test_kl  : KL divergence on the held-out test set (Bernoulli)
test_mse : MSE in PSI space on the held-out test set
sumdiff_w: model.energy_seq_struct.w (scalar learnable weight)
sumdiff_b: model.energy_seq_struct.b (scalar learnable bias)

Usage:
  python train_monitored.py \
      --train-npz data/train_data.npz \
      --test-npz  data/test_data.npz \
      --input-length 90 \
      --checkpoint-dir checkpoints/monitored_seed42 \
      --staged \
      --stage1-epochs 50 --stage2-epochs 50 --epochs 100 \
      --stage4-epochs 100 --stage5-epochs 100 \
      --patience 15 --stage5-patience 10 \
      --lr 1e-3 --stage4-lr 1e-4 --stage5-lr 1e-4 \
      --l1-lambda 2e-3 --l2-smooth 1e-3 \
      --no-batchnorm --constant-var \
      --seed 42 \
      --csv-log metrics_seed42.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from model import PNASModel
from train import (
    PSIDataset,
    kl_divergence_from_logits,
    gaussian_nll_const_var,
    gaussian_nll_learned_var,
    _freeze_for_stage,
    _init_variance_to_one,
    _l2_smooth_loss,
    rmse,
    train_epoch,
    build_parser,
)

logger = logging.getLogger(__name__)


# ── Test-set evaluation ───────────────────────────────────────────────────────

def eval_test(model, loader, device):
    """Return test KL divergence and PSI-space MSE (no regularization)."""
    model.eval()
    kl_sum = mse_sum = 0.0
    n = 0
    with torch.no_grad():
        for x_seq, x_struct, x_wobble, y in loader:
            x_seq    = x_seq.to(device)
            x_struct = x_struct.to(device)
            x_wobble = x_wobble.to(device)
            y        = y.to(device)
            mask     = y.isfinite()
            if mask.sum() == 0:
                continue

            if model.stage >= 4:
                logits, _, _ = model(x_seq, x_struct, x_wobble,
                                     return_logits=True, return_uncertainty=True)
            else:
                logits = model(x_seq, x_struct, x_wobble, return_logits=True)

            logits_m = logits[mask]
            y_m      = y[mask]
            cnt      = mask.sum().item()

            kl_sum  += kl_divergence_from_logits(logits_m, y_m).item() * cnt
            mse_sum += F.mse_loss(torch.sigmoid(logits_m), y_m).item() * cnt
            n       += cnt

    return {"test_kl": kl_sum / n, "test_mse": mse_sum / n}


def _sumdiff_params(model):
    w = model.energy_seq_struct.w.item()
    b = model.energy_seq_struct.b.item()
    return w, b


# ── Monitored single-stage loop ───────────────────────────────────────────────

def _train_one_stage_monitored(
    model, train_loader, val_loader, test_loader, hparams, csv_writer, stage, seed
):
    device         = hparams["device"]
    num_epochs     = hparams["num_epochs"]
    patience       = hparams["patience"]
    checkpoint_dir = hparams["checkpoint_dir"]
    lr             = hparams["lr"]
    weight_decay   = hparams.get("weight_decay", 0.0)
    loss_fn        = hparams.get("loss_fn", kl_divergence_from_logits)
    l1_lambda      = hparams.get("l1_lambda", 0.0)
    l2_smooth      = hparams.get("l2_smooth", 0.0)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp       = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(checkpoint_dir, f"best_model_{timestamp}.pt")

    logger.info(f"  Checkpoint → {checkpoint_path}")
    logger.info(f"  Max epochs: {num_epochs}  Patience: {patience}  LR: {lr}")

    best_val_loss              = float("inf")
    epochs_without_improvement = 0
    history                    = []

    for epoch in range(1, num_epochs + 1):
        logger.info(f"  Epoch {epoch}/{num_epochs}")

        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device,
                                    l1_lambda=l1_lambda, l2_smooth=l2_smooth)
        val_metrics   = eval_test(model, val_loader, device)   # reuse eval_test for val
        test_metrics  = eval_test(model, test_loader, device)
        sd_w, sd_b    = _sumdiff_params(model)

        record = {
            "epoch":      epoch,
            "train_loss": train_metrics["loss"],
            "train_rmse": train_metrics["rmse"],
            "val_loss":   val_metrics["test_kl"],   # val KL used for early stopping
            "val_rmse":   val_metrics["test_mse"],
        }
        history.append(record)

        logger.info(
            f"    train loss={train_metrics['loss']:.6f}  rmse={train_metrics['rmse']:.6f}"
        )
        logger.info(
            f"    val   kl={val_metrics['test_kl']:.6f}  mse={val_metrics['test_mse']:.6f}"
        )
        logger.info(
            f"    test  kl={test_metrics['test_kl']:.6f}  mse={test_metrics['test_mse']:.6f}"
            f"  sumdiff w={sd_w:.6f}  b={sd_b:.6f}"
        )

        # Write CSV row
        csv_writer.writerow({
            "seed":       seed,
            "stage":      stage,
            "epoch":      epoch,
            "train_loss": f"{train_metrics['loss']:.8f}",
            "val_kl":     f"{val_metrics['test_kl']:.8f}",
            "val_mse":    f"{val_metrics['test_mse']:.8f}",
            "test_kl":    f"{test_metrics['test_kl']:.8f}",
            "test_mse":   f"{test_metrics['test_mse']:.8f}",
            "sumdiff_w":  f"{sd_w:.8f}",
            "sumdiff_b":  f"{sd_b:.8f}",
        })

        val_loss = val_metrics["test_kl"]
        if val_loss < best_val_loss:
            prev_best     = best_val_loss
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch":                epoch,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss":        best_val_loss,
                    "history":              history,
                    "hparams":              {k: str(v) for k, v in hparams.items()},
                },
                checkpoint_path,
            )
            logger.info(f"    ✓ val kl {prev_best:.6f} → {best_val_loss:.6f}  saved")
        else:
            epochs_without_improvement += 1
            logger.info(
                f"    no improvement {epochs_without_improvement}/{patience} "
                f"(best={best_val_loss:.6f})"
            )
            if epochs_without_improvement >= patience:
                logger.info(f"  Early stopping after {epoch} epochs.")
                break

    logger.info(f"  Stage complete — best val kl: {best_val_loss:.6f}")
    return {"history": history, "best_val_loss": best_val_loss,
            "checkpoint_path": checkpoint_path}


# ── Monitored 5-stage training ────────────────────────────────────────────────

def train_staged_monitored(
    model, train_loader, val_loader, test_loader, hparams, csv_writer, seed
):
    device         = hparams["device"]
    checkpoint_dir = hparams["checkpoint_dir"]
    stage4_epochs  = hparams.get("stage4_epochs", 0)
    stage4_lr      = hparams.get("stage4_lr", hparams["lr"] * 0.1)
    stage_epochs   = [
        hparams.get("stage1_epochs", 50),
        hparams.get("stage2_epochs", 50),
        hparams.get("stage3_epochs", hparams["num_epochs"]),
    ]

    model = model.to(device)
    results = {}

    for stage in [1, 2, 3]:
        logger.info(f"\n{'='*60}")
        logger.info(f"=== STAGE {stage} — {stage_epochs[stage-1]} epochs ===")
        logger.info(f"{'='*60}")

        model.stage = stage
        _freeze_for_stage(model, stage)

        if stage > 1:
            prev_ckpt = results[stage - 1]["checkpoint_path"]
            logger.info(f"  Loading stage {stage-1} checkpoint: {prev_ckpt}")
            ckpt = torch.load(prev_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            _freeze_for_stage(model, stage)

        stage_hparams = {
            **hparams,
            "num_epochs":     stage_epochs[stage - 1],
            "checkpoint_dir": os.path.join(checkpoint_dir, f"stage{stage}"),
        }
        results[stage] = _train_one_stage_monitored(
            model, train_loader, val_loader, test_loader,
            stage_hparams, csv_writer, stage, seed
        )

    # Stage 4
    if stage4_epochs > 0:
        logger.info(f"\n{'='*60}")
        logger.info(f"=== STAGE 4 — {stage4_epochs} epochs (variance fine-tuning) ===")
        logger.info(f"{'='*60}")

        model.stage = 4
        _freeze_for_stage(model, 4)

        prev_ckpt = results[3]["checkpoint_path"]
        ckpt = torch.load(prev_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        _freeze_for_stage(model, 4)
        _init_variance_to_one(model)

        stage4_hparams = {
            **hparams,
            "num_epochs":     stage4_epochs,
            "lr":             stage4_lr,
            "checkpoint_dir": os.path.join(checkpoint_dir, "stage4"),
        }
        results[4] = _train_one_stage_monitored(
            model, train_loader, val_loader, test_loader,
            stage4_hparams, csv_writer, 4, seed
        )

    # Stage 5
    stage5_epochs   = hparams.get("stage5_epochs", 0)
    stage5_lr       = hparams.get("stage5_lr", stage4_lr)
    stage5_patience = hparams.get("stage5_patience", 10)

    if stage5_epochs > 0:
        if stage4_epochs == 0:
            raise ValueError("Stage 5 requires stage 4.")
        logger.info(f"\n{'='*60}")
        logger.info(f"=== STAGE 5 — {stage5_epochs} epochs (full joint fine-tuning) ===")
        logger.info(f"{'='*60}")

        model.stage = 5
        _freeze_for_stage(model, 5)

        prev_ckpt = results[4]["checkpoint_path"]
        ckpt = torch.load(prev_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        _freeze_for_stage(model, 5)

        stage5_hparams = {
            **hparams,
            "num_epochs":     stage5_epochs,
            "lr":             stage5_lr,
            "patience":       stage5_patience,
            "checkpoint_dir": os.path.join(checkpoint_dir, "stage5"),
        }
        results[5] = _train_one_stage_monitored(
            model, train_loader, val_loader, test_loader,
            stage5_hparams, csv_writer, 5, seed
        )

    final = 5 if stage5_epochs > 0 else (4 if stage4_epochs > 0 else 3)
    model.stage = final
    logger.info("\n=== Staged training complete ===")
    for s in [1, 2, 3, 4, 5]:
        if s in results:
            logger.info(f"  Stage {s} best val kl: {results[s]['best_val_loss']:.6f}")
    return results[final]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    parser.add_argument(
        "--csv-log", default="training_metrics.csv", dest="csv_log",
        help="Path to per-epoch CSV log file.",
    )
    parser.add_argument(
        "--test-npz", default=None, dest="test_npz",
        help="Held-out test set for per-epoch evaluation.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Device: {device}")

    # ── Training data ─────────────────────────────────────────────────────────
    train_npz = np.load(args.train_npz)
    dataset   = PSIDataset(
        train_npz["seq_oh"], train_npz["struct_oh"],
        train_npz["wobbles"], train_npz["metadata_PSI"].astype(np.float32),
    )
    n_val   = int(args.val_split * len(dataset))
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    logger.info(f"Train: {n_train:,}  Val: {n_val:,}")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True,  pin_memory=pin)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, pin_memory=pin)

    # ── Test data ─────────────────────────────────────────────────────────────
    if args.test_npz is None:
        raise ValueError("--test-npz is required for train_monitored.py")
    test_npz  = np.load(args.test_npz)
    test_ds   = PSIDataset(
        test_npz["seq_oh"], test_npz["struct_oh"],
        test_npz["wobbles"], test_npz["metadata_PSI"].astype(np.float32),
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, pin_memory=pin)
    logger.info(f"Test:  {len(test_ds):,}")

    # ── Model ─────────────────────────────────────────────────────────────────
    use_batchnorm = not args.no_batchnorm
    model = PNASModel(input_length=args.input_length, use_batchnorm=use_batchnorm)

    loss_fn = gaussian_nll_const_var if args.constant_var else kl_divergence_from_logits

    hparams = {
        "device":           device,
        "num_epochs":       args.epochs,
        "patience":         args.patience,
        "checkpoint_dir":   args.checkpoint_dir,
        "lr":               args.lr,
        "weight_decay":     args.weight_decay,
        "loss_fn":          loss_fn,
        "l1_lambda":        args.l1_lambda,
        "l2_smooth":        args.l2_smooth,
        "stage1_epochs":    args.stage1_epochs,
        "stage2_epochs":    args.stage2_epochs,
        "stage3_epochs":    args.epochs,
        "stage4_epochs":    args.stage4_epochs,
        "stage4_lr":        args.stage4_lr if args.stage4_lr is not None else args.lr * 0.1,
        "stage5_epochs":    args.stage5_epochs,
        "stage5_lr":        args.stage5_lr if args.stage5_lr is not None else
                            (args.stage4_lr if args.stage4_lr is not None else args.lr * 0.1),
        "stage5_patience":  args.stage5_patience,
    }

    # ── CSV setup ─────────────────────────────────────────────────────────────
    csv_fields = ["seed", "stage", "epoch", "train_loss",
                  "val_kl", "val_mse", "test_kl", "test_mse",
                  "sumdiff_w", "sumdiff_b"]
    os.makedirs(os.path.dirname(os.path.abspath(args.csv_log)), exist_ok=True)
    csv_fh = open(args.csv_log, "w", newline="")
    csv_writer = csv.DictWriter(csv_fh, fieldnames=csv_fields)
    csv_writer.writeheader()
    csv_fh.flush()

    logger.info(f"CSV log → {args.csv_log}")

    try:
        train_staged_monitored(
            model, train_loader, val_loader, test_loader,
            hparams, csv_writer, seed=args.seed,
        )
    finally:
        csv_fh.flush()
        csv_fh.close()
        logger.info(f"CSV log closed: {args.csv_log}")


if __name__ == "__main__":
    main()
