"""Training script for the PNAS splicing model."""

from __future__ import annotations

import argparse
import logging
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm.auto import tqdm

from model import PNASModel

logger = logging.getLogger(__name__)


# ── Dataset ───────────────────────────────────────────────────────────────────

class PSIDataset(Dataset):
    def __init__(self, x_seq, x_struct, x_wobble, y):
        self.x_seq    = torch.as_tensor(x_seq)
        self.x_struct = torch.as_tensor(x_struct)
        self.x_wobble = torch.as_tensor(x_wobble)
        self.y        = torch.as_tensor(y, dtype=torch.float32)

        n = len(self.y)
        if not (len(self.x_seq) == len(self.x_struct) == len(self.x_wobble) == n):
            raise ValueError("All inputs and labels must have the same number of samples.")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_seq[idx], self.x_struct[idx], self.x_wobble[idx], self.y[idx]


# ── Loss and metrics ──────────────────────────────────────────────────────────

def kl_divergence_from_logits(logits, targets):
    """KL divergence between two Bernoullis: one from logits, one from targets."""
    return (
        F.binary_cross_entropy_with_logits(logits, targets)
        - F.binary_cross_entropy(targets, targets)
    )


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def _l2_smooth_loss(model: torch.nn.Module) -> torch.Tensor:
    """Squared first differences on position bias weights (L2 smoothness)."""
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, p in model.named_parameters():
        if "position_bias" in name and p.numel() > 1:
            loss = loss + ((p[1:] - p[:-1]) ** 2).sum()
    return loss


# ── Per-epoch loops ───────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, loss_fn, device,
                l1_lambda: float = 0.0, l2_smooth: float = 0.0) -> dict:
    """One forward+backward pass. Supports L1 activity and L2 smoothness reg."""
    model.train()
    total_loss = total_l1 = total_l2 = 0.0
    pred_list, target_list = [], []

    # Hook captures post-softplus activations for L1 (seq-only in stage 1,
    # seq+struct in stages 2/3 — correct either way since hooks attach to the
    # energy_activation modules which see whatever is concatenated).
    _act = [None, None]
    hooks = []
    if l1_lambda > 0.0:
        hooks.append(model.energy_activation_incl.register_forward_hook(
            lambda m, inp, out: _act.__setitem__(0, out)
        ))
        hooks.append(model.energy_activation_skip.register_forward_hook(
            lambda m, inp, out: _act.__setitem__(1, out)
        ))

    try:
        pbar = tqdm(loader, desc="  train", leave=False, unit="batch")
        for seq, struct, wobble, y in pbar:
            seq    = seq.to(device, non_blocking=True)
            struct = struct.to(device, non_blocking=True)
            wobble = wobble.to(device, non_blocking=True)
            y      = y.to(device, dtype=torch.float32, non_blocking=True)

            optimizer.zero_grad()
            logits     = model(seq, struct, wobble, return_logits=True)
            pred_probs = torch.sigmoid(logits)
            loss       = loss_fn(logits, y)

            if l1_lambda > 0.0:
                # sum over (filters, positions) per example, then mean over batch
                # — keeps gradient per activation at 1/B rather than 1/(B*F*P)
                l1_reg   = l1_lambda * (
                    _act[0].sum(dim=(1, 2)).mean() +
                    _act[1].sum(dim=(1, 2)).mean()
                )
                loss     = loss + l1_reg
                total_l1 += l1_reg.item() * y.size(0)

            if l2_smooth > 0.0:
                l2_reg   = l2_smooth * _l2_smooth_loss(model)
                loss     = loss + l2_reg
                total_l2 += l2_reg.item() * y.size(0)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            pred_list.append(pred_probs.detach())
            target_list.append(y.detach())
            pbar.set_postfix(batch_loss=f"{loss.item():.5f}")
    finally:
        for h in hooks:
            h.remove()

    n     = len(loader.dataset)
    preds = torch.cat(pred_list)
    tgts  = torch.cat(target_list)
    out   = {"loss": total_loss / n, "rmse": rmse(preds, tgts)}
    if l1_lambda > 0.0:
        out["l1"] = total_l1 / n
    if l2_smooth > 0.0:
        out["l2"] = total_l2 / n
    return out


def eval_epoch(model, loader, loss_fn, device) -> dict:
    """No-grad evaluation pass. Returns pure KL loss (no regularization terms)."""
    model.eval()
    total_loss = 0.0
    pred_list, target_list = [], []

    pbar = tqdm(loader, desc="  eval ", leave=False, unit="batch")
    with torch.no_grad():
        for seq, struct, wobble, y in pbar:
            seq    = seq.to(device, non_blocking=True)
            struct = struct.to(device, non_blocking=True)
            wobble = wobble.to(device, non_blocking=True)
            y      = y.to(device, dtype=torch.float32, non_blocking=True)

            logits     = model(seq, struct, wobble, return_logits=True)
            pred_probs = torch.sigmoid(logits)
            loss       = loss_fn(logits, y)

            total_loss += loss.item() * y.size(0)
            pred_list.append(pred_probs)
            target_list.append(y)

    preds   = torch.cat(pred_list)
    targets = torch.cat(target_list)
    return {"loss": total_loss / len(loader.dataset), "rmse": rmse(preds, targets)}


# ── Freeze helpers ────────────────────────────────────────────────────────────

_STRUCT_PARAMS = {"conv_struct_incl", "conv_struct_skip",
                  "position_bias_incl_struct", "position_bias_skip_struct"}
_TUNER_PARAMS  = {"tuner", "variance_bottleneck", "variance_tuner"}


def _freeze_for_stage(model: torch.nn.Module, stage: int) -> None:
    """Set requires_grad on each parameter according to the training stage.

    Stage 1: only seq conv + seq position bias + SumDiff trainable.
    Stage 2: add struct conv + struct position bias.
    Stage 3: everything except variance branch (which is not used here).
    """
    frozen_groups = _TUNER_PARAMS.copy()       # always freeze tuner in staged training
    if stage == 1:
        frozen_groups |= _STRUCT_PARAMS        # also freeze struct in stage 1

    for name, p in model.named_parameters():
        is_frozen = any(name.startswith(g) or g in name for g in frozen_groups)
        p.requires_grad = not is_frozen

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"  Stage {stage} — trainable params: {n_train:,} / {n_total:,}")


# ── Single-stage training loop ────────────────────────────────────────────────

def _train_one_stage(model, train_loader, val_loader, hparams: dict) -> dict:
    """Standard early-stopping loop. Optimizer built from requires_grad params only."""
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
        val_metrics   = eval_epoch(model, val_loader, loss_fn, device)

        record = {
            "epoch":      epoch,
            "train_loss": train_metrics["loss"],
            "train_rmse": train_metrics["rmse"],
            "val_loss":   val_metrics["loss"],
            "val_rmse":   val_metrics["rmse"],
        }
        history.append(record)

        logger.info(
            f"    train loss={train_metrics['loss']:.6f}  rmse={train_metrics['rmse']:.6f}"
        )
        logger.info(
            f"    val   loss={val_metrics['loss']:.6f}  rmse={val_metrics['rmse']:.6f}"
        )

        if val_metrics["loss"] < best_val_loss:
            prev_best     = best_val_loss
            best_val_loss = val_metrics["loss"]
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
            logger.info(f"    ✓ val loss {prev_best:.6f} → {best_val_loss:.6f}  saved")
        else:
            epochs_without_improvement += 1
            logger.info(
                f"    no improvement {epochs_without_improvement}/{patience} "
                f"(best={best_val_loss:.6f})"
            )
            if epochs_without_improvement >= patience:
                logger.info(f"  Early stopping after {epoch} epochs.")
                break

    logger.info(f"  Stage complete — best val loss: {best_val_loss:.6f}")
    return {"history": history, "best_val_loss": best_val_loss,
            "checkpoint_path": checkpoint_path}


# ── Staged training (3-stage PNAS schedule) ───────────────────────────────────

def train_staged(model, train_loader, val_loader, hparams: dict) -> dict:
    """Three-stage training matching the original PNAS paper schedule.

    Stage 1: seq filters + SumDiff only, struct/tuner frozen.
    Stage 2: seq + struct filters, tuner still frozen.
    Stage 3: full model including ResidualTuner.

    L1/L2 regularization is applied throughout all stages.
    """
    device         = hparams["device"]
    checkpoint_dir = hparams["checkpoint_dir"]
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

        # Load best checkpoint from previous stage (skip on stage 1)
        if stage > 1:
            prev_ckpt = results[stage - 1]["checkpoint_path"]
            logger.info(f"  Loading stage {stage-1} checkpoint: {prev_ckpt}")
            ckpt = torch.load(prev_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            # Re-apply freeze after load_state_dict (it doesn't touch requires_grad)
            _freeze_for_stage(model, stage)

        stage_hparams = {
            **hparams,
            "num_epochs":     stage_epochs[stage - 1],
            "checkpoint_dir": os.path.join(checkpoint_dir, f"stage{stage}"),
        }
        results[stage] = _train_one_stage(model, train_loader, val_loader, stage_hparams)

    # Reset to stage 3 for inference
    model.stage = 3
    logger.info("\n=== Staged training complete ===")
    logger.info(f"  Stage 1 best val loss: {results[1]['best_val_loss']:.6f}")
    logger.info(f"  Stage 2 best val loss: {results[2]['best_val_loss']:.6f}")
    logger.info(f"  Stage 3 best val loss: {results[3]['best_val_loss']:.6f}")
    return results[3]


# ── Non-staged training loop (unchanged from Arush's version) ────────────────

def train(model, train_loader, val_loader, hparams: dict) -> dict:
    """Standard single-stage training loop with optional L1/L2 regularization."""
    device         = hparams["device"]
    num_epochs     = hparams["num_epochs"]
    patience       = hparams["patience"]
    checkpoint_dir = hparams["checkpoint_dir"]
    lr             = hparams["lr"]
    weight_decay   = hparams.get("weight_decay", 0.0)
    loss_fn        = hparams.get("loss_fn", kl_divergence_from_logits)
    l1_lambda      = hparams.get("l1_lambda", 0.0)
    l2_smooth      = hparams.get("l2_smooth", 0.0)

    model     = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp       = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(checkpoint_dir, f"best_model_{timestamp}.pt")

    logger.info("=== Training configuration ===")
    logger.info(f"  Device:          {device}")
    logger.info(f"  Max epochs:      {num_epochs}")
    logger.info(f"  Patience:        {patience}")
    logger.info(f"  LR:              {lr}")
    logger.info(f"  Weight decay:    {weight_decay}")
    logger.info(f"  L1 lambda:       {l1_lambda}")
    logger.info(f"  L2 smooth:       {l2_smooth}")
    logger.info(f"  Train batches:   {len(train_loader)}")
    logger.info(f"  Val batches:     {len(val_loader)}")
    logger.info(f"  Checkpoint path: {checkpoint_path}")

    best_val_loss              = float("inf")
    epochs_without_improvement = 0
    history                    = []

    for epoch in range(1, num_epochs + 1):
        logger.info(f"--- Epoch {epoch}/{num_epochs} ---")

        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device,
                                    l1_lambda=l1_lambda, l2_smooth=l2_smooth)
        val_metrics   = eval_epoch(model, val_loader, loss_fn, device)

        record = {
            "epoch":      epoch,
            "train_loss": train_metrics["loss"],
            "train_rmse": train_metrics["rmse"],
            "val_loss":   val_metrics["loss"],
            "val_rmse":   val_metrics["rmse"],
        }
        history.append(record)

        logger.info(
            f"  Train — loss: {train_metrics['loss']:.6f}  rmse: {train_metrics['rmse']:.6f}"
        )
        logger.info(
            f"  Val   — loss: {val_metrics['loss']:.6f}  rmse: {val_metrics['rmse']:.6f}"
        )

        if val_metrics["loss"] < best_val_loss:
            prev_best     = best_val_loss
            best_val_loss = val_metrics["loss"]
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
            logger.info(
                f"  Val loss improved: {prev_best:.6f} -> {best_val_loss:.6f}"
                f" — checkpoint saved"
            )
        else:
            epochs_without_improvement += 1
            logger.info(
                f"  No improvement: {epochs_without_improvement}/{patience} "
                f"(best val loss so far: {best_val_loss:.6f})"
            )
            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs.")
                break

    logger.info(f"=== Training complete — best val loss: {best_val_loss:.6f} ===")
    return {
        "history":         history,
        "best_val_loss":   best_val_loss,
        "checkpoint_path": checkpoint_path,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the PNAS splicing model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("data")
    data.add_argument("--train-npz", required=True)
    data.add_argument("--test-npz",  default=None)
    data.add_argument("--val-split", type=float, default=0.1)

    mdl = parser.add_argument_group("model")
    mdl.add_argument("--input-length", type=int, default=90)
    mdl.add_argument("--no-batchnorm", action="store_true")
    mdl.add_argument(
        "--checkpoint", default=None, metavar="PATH",
        help="Warm-start from this checkpoint (partial load supported).",
    )

    opt = parser.add_argument_group("optimization")
    opt.add_argument("--batch-size",   type=int,   default=64)
    opt.add_argument("--epochs",       type=int,   default=100,
                     help="Max epochs per stage (or total for non-staged).")
    opt.add_argument("--lr",           type=float, default=1e-3)
    opt.add_argument("--weight-decay", type=float, default=0.0)
    opt.add_argument("--patience",     type=int,   default=10)
    opt.add_argument("--l1-lambda",    type=float, default=0.0, dest="l1_lambda",
                     help="L1 activity regularization on post-softplus filter activations.")
    opt.add_argument("--l2-smooth",    type=float, default=0.0, dest="l2_smooth",
                     help="L2 smoothness regularization on position bias weights.")

    stg = parser.add_argument_group("staged training")
    stg.add_argument("--staged", action="store_true",
                     help="Use 3-stage training schedule from the PNAS paper.")
    stg.add_argument("--stage1-epochs", type=int, default=50, dest="stage1_epochs",
                     help="Epochs for stage 1 (seq only, simplified tuner).")
    stg.add_argument("--stage2-epochs", type=int, default=50, dest="stage2_epochs",
                     help="Epochs for stage 2 (seq+struct, simplified tuner).")

    run = parser.add_argument_group("runtime")
    run.add_argument("--checkpoint-dir", default="./checkpoints")
    run.add_argument("--device", default=None)
    run.add_argument("--seed",   type=int, default=42)

    return parser


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

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

    logger.info(f"Loading training data: {args.train_npz}")
    train_npz = np.load(args.train_npz)
    x_seq     = train_npz["seq_oh"]
    x_struct  = train_npz["struct_oh"]
    x_wobble  = train_npz["wobbles"]
    y         = train_npz["metadata_PSI"].astype(np.float32)
    logger.info(
        f"  Loaded — examples: {len(y):,}, "
        f"seq: {x_seq.shape}, struct: {x_struct.shape}, wobble: {x_wobble.shape}"
    )

    dataset = PSIDataset(x_seq, x_struct, x_wobble, y)
    n_val   = int(args.val_split * len(dataset))
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    logger.info(f"Split — train: {n_train:,}  val: {n_val:,}")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True,  pin_memory=pin)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, pin_memory=pin)
    logger.info(
        f"DataLoaders — train: {len(train_loader)} batches, "
        f"val: {len(val_loader)} batches (batch_size={args.batch_size})"
    )

    use_batchnorm = not args.no_batchnorm
    logger.info(f"Instantiating PNASModel — input_length={args.input_length}, "
                f"use_batchnorm={use_batchnorm}")
    model = PNASModel(input_length=args.input_length, use_batchnorm=use_batchnorm)

    if args.checkpoint is not None:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        raw        = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state_dict = raw.get("model_state_dict", raw)
        model.load_partial_state_dict(state_dict)
    else:
        logger.info("No checkpoint — training from random initialization.")

    hparams = {
        "device":         device,
        "num_epochs":     args.epochs,
        "patience":       args.patience,
        "checkpoint_dir": args.checkpoint_dir,
        "lr":             args.lr,
        "weight_decay":   args.weight_decay,
        "loss_fn":        kl_divergence_from_logits,
        "l1_lambda":      args.l1_lambda,
        "l2_smooth":      args.l2_smooth,
        "stage1_epochs":  args.stage1_epochs,
        "stage2_epochs":  args.stage2_epochs,
        "stage3_epochs":  args.epochs,
    }

    if args.staged:
        results = train_staged(model, train_loader, val_loader, hparams)
    else:
        results = train(model, train_loader, val_loader, hparams)

    if args.test_npz is not None:
        logger.info(f"Loading test data: {args.test_npz}")
        test_npz     = np.load(args.test_npz)
        test_dataset = PSIDataset(
            test_npz["seq_oh"], test_npz["struct_oh"],
            test_npz["wobbles"], test_npz["metadata_PSI"].astype(np.float32),
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, pin_memory=pin)
        logger.info(f"  Test — examples: {len(test_dataset):,}")

        logger.info(f"Reloading best checkpoint: {results['checkpoint_path']}")
        best_ckpt = torch.load(results["checkpoint_path"], map_location=device,
                               weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        model.stage = 3
        model = model.to(device)

        test_metrics = eval_epoch(model, test_loader, kl_divergence_from_logits, device)
        logger.info(
            f"Test — loss: {test_metrics['loss']:.6f}  rmse: {test_metrics['rmse']:.6f}"
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
