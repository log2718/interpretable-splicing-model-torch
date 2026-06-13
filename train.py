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


def gaussian_nll_logit(logit_mu, log_var, var, targets, lambda_: float = 1.0):
    """Gaussian NLL in logit space with a tunable regularization weight.

    Loss = mean(lambda_ * 0.5 * log_var  +  0.5 * (logit_true - logit_mu)² / var)

    Targets are clamped to [1e-2, 1-1e-2] before logit to avoid penalising
    unresolvable differences at extreme PSI (e.g. 99% vs 99.9%).

    Args:
        logit_mu: Predicted mean in logit space, shape ``(batch,)``.
        log_var:  log of predicted variance, shape ``(batch,)``.
        var:      Predicted variance (softplus + eps), shape ``(batch,)``.
        targets:  True PSI values in [0, 1], shape ``(batch,)``.
        lambda_:  Weight on the log_var term (uncertainty regularisation).

    Returns:
        Tuple ``(loss, term1_val, term2_val)`` where ``loss`` is the scalar
        mean NLL (with gradient), and ``term1_val`` / ``term2_val`` are
        Python floats for logging (lambda_ * log_var term and residual term).
    """
    logit_true = torch.logit(targets.clamp(1e-2, 1 - 1e-2))
    term1 = 0.5 * log_var                               # uncertainty penalty
    term2 = 0.5 * (logit_true - logit_mu) ** 2 / var   # fit term
    loss  = (lambda_ * term1 + term2).mean()
    return loss, (lambda_ * term1).mean().item(), term2.mean().item()


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


# ── Per-epoch loops ───────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, loss_fn, device,
                uncertainty: bool = False, lambda_: float = 1.0) -> dict:
    """One forward+backward pass over ``loader``. Returns loss, RMSE, and
    (when uncertainty=True) per-term NLL components and mean KL divergence."""
    model.train()
    total_loss = total_term1 = total_term2 = total_kl = 0.0
    pred_list, target_list = [], []

    pbar = tqdm(loader, desc="  train", leave=False, unit="batch")
    for seq, struct, wobble, y in pbar:
        seq    = seq.to(device, non_blocking=True)
        struct = struct.to(device, non_blocking=True)
        wobble = wobble.to(device, non_blocking=True)
        y      = y.to(device, dtype=torch.float32, non_blocking=True)

        optimizer.zero_grad()

        if uncertainty:
            logit_mu, log_var, var = model(seq, struct, wobble,
                                           return_logits=True,
                                           return_uncertainty=True)
            pred_probs = torch.sigmoid(logit_mu)
            loss, t1, t2 = gaussian_nll_logit(logit_mu, log_var, var, y, lambda_)
            total_term1 += t1 * y.size(0)
            total_term2 += t2 * y.size(0)
            with torch.no_grad():
                total_kl += kl_divergence_from_logits(logit_mu, y).item() * y.size(0)
        else:
            logits     = model(seq, struct, wobble, return_logits=True)
            pred_probs = torch.sigmoid(logits)
            loss       = loss_fn(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        pred_list.append(pred_probs.detach())
        target_list.append(y.detach())

        pbar.set_postfix(batch_loss=f"{loss.item():.5f}")

    n     = len(loader.dataset)
    preds = torch.cat(pred_list)
    tgts  = torch.cat(target_list)
    out   = {"loss": total_loss / n, "rmse": rmse(preds, tgts)}
    if uncertainty:
        out["term1"] = total_term1 / n
        out["term2"] = total_term2 / n
        out["kl"]    = total_kl    / n
    return out


def eval_epoch(model, loader, loss_fn, device,
               uncertainty: bool = False, lambda_: float = 1.0) -> dict:
    """No-grad evaluation pass over ``loader``. Returns loss, RMSE, and
    (when uncertainty=True) per-term NLL components and mean KL divergence."""
    model.eval()
    total_loss = total_term1 = total_term2 = total_kl = 0.0
    pred_list, target_list = [], []

    pbar = tqdm(loader, desc="  eval ", leave=False, unit="batch")
    with torch.no_grad():
        for seq, struct, wobble, y in pbar:
            seq    = seq.to(device, non_blocking=True)
            struct = struct.to(device, non_blocking=True)
            wobble = wobble.to(device, non_blocking=True)
            y      = y.to(device, dtype=torch.float32, non_blocking=True)

            if uncertainty:
                logit_mu, log_var, var = model(seq, struct, wobble,
                                               return_logits=True,
                                               return_uncertainty=True)
                pred_probs = torch.sigmoid(logit_mu)
                loss, t1, t2 = gaussian_nll_logit(logit_mu, log_var, var, y, lambda_)
                total_term1 += t1 * y.size(0)
                total_term2 += t2 * y.size(0)
                total_kl    += kl_divergence_from_logits(logit_mu, y).item() * y.size(0)
            else:
                logits     = model(seq, struct, wobble, return_logits=True)
                pred_probs = torch.sigmoid(logits)
                loss       = loss_fn(logits, y)

            total_loss += loss.item() * y.size(0)
            pred_list.append(pred_probs)
            target_list.append(y)

            pbar.set_postfix(batch_loss=f"{loss.item():.5f}")

    n     = len(loader.dataset)
    preds = torch.cat(pred_list)
    tgts  = torch.cat(target_list)
    out   = {"loss": total_loss / n, "rmse": rmse(preds, tgts)}
    if uncertainty:
        out["term1"] = total_term1 / n
        out["term2"] = total_term2 / n
        out["kl"]    = total_kl    / n
    return out


# ── Training loop ─────────────────────────────────────────────────────────────

def train(model, train_loader, val_loader, hparams: dict) -> dict:
    """Outer training loop with early stopping and checkpointing.

    Args:
        model: PNASModel instance.
        train_loader: DataLoader for training examples.
        val_loader: DataLoader for validation examples.
        hparams: Dict with keys: device, num_epochs, patience, checkpoint_dir,
                 lr, weight_decay, loss_fn, uncertainty, freeze_epochs.

    Returns:
        Dict with keys: history (list of per-epoch dicts), best_val_loss (float),
        checkpoint_path (str).
    """
    device         = hparams["device"]
    num_epochs     = hparams["num_epochs"]
    patience       = hparams["patience"]
    checkpoint_dir = hparams["checkpoint_dir"]
    lr             = hparams["lr"]
    weight_decay   = hparams.get("weight_decay", 0.0)
    loss_fn        = hparams.get("loss_fn", kl_divergence_from_logits)
    uncertainty    = hparams.get("uncertainty", False)
    freeze_epochs  = hparams.get("freeze_epochs", 0)
    lambda_        = hparams.get("lambda_", 1.0)

    model = model.to(device)

    # Phase 1: freeze all params except the variance branch
    if uncertainty and freeze_epochs > 0:
        for name, p in model.named_parameters():
            p.requires_grad = ("variance_bottleneck" in name or "variance_tuner" in name)
        logger.info(
            f"Phase 1 — freezing all parameters except variance_bottleneck / variance_tuner "
            f"for {freeze_epochs} epoch(s)."
        )

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(checkpoint_dir, f"best_model_{timestamp}.pt")

    logger.info("=== Training configuration ===")
    logger.info(f"  Device:          {device}")
    logger.info(f"  Max epochs:      {num_epochs}")
    logger.info(f"  Patience:        {patience}")
    logger.info(f"  LR:              {lr}")
    logger.info(f"  Weight decay:    {weight_decay}")
    logger.info(f"  Train batches:   {len(train_loader)}")
    logger.info(f"  Val batches:     {len(val_loader)}")
    logger.info(f"  Checkpoint path: {checkpoint_path}")

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, num_epochs + 1):
        logger.info(f"--- Epoch {epoch}/{num_epochs} ---")

        # Phase 2: unfreeze all parameters after freeze_epochs
        if uncertainty and freeze_epochs > 0 and epoch == freeze_epochs + 1:
            logger.info(
                f"Phase 2 — unfreezing all parameters, rebuilding optimizer "
                f"with lr={hparams.get('lr_phase2', lr * 0.1):.2e}"
            )
            for p in model.parameters():
                p.requires_grad = True
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=hparams.get("lr_phase2", lr * 0.1),
                weight_decay=weight_decay,
            )

        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device,
                                    uncertainty=uncertainty, lambda_=lambda_)
        val_metrics   = eval_epoch(model, val_loader, loss_fn, device,
                                   uncertainty=uncertainty, lambda_=lambda_)

        record = {
            "epoch":      epoch,
            "train_loss": train_metrics["loss"],
            "train_rmse": train_metrics["rmse"],
            "val_loss":   val_metrics["loss"],
            "val_rmse":   val_metrics["rmse"],
        }
        if uncertainty:
            record.update({
                "train_term1": train_metrics["term1"],
                "train_term2": train_metrics["term2"],
                "train_kl":    train_metrics["kl"],
                "val_term1":   val_metrics["term1"],
                "val_term2":   val_metrics["term2"],
                "val_kl":      val_metrics["kl"],
            })
        history.append(record)

        logger.info(
            f"  Train — loss: {train_metrics['loss']:.6f}  rmse: {train_metrics['rmse']:.6f}"
        )
        logger.info(
            f"  Val   — loss: {val_metrics['loss']:.6f}  rmse: {val_metrics['rmse']:.6f}"
        )
        if uncertainty:
            logger.info(
                f"  Val   — term1(λ·log_var): {val_metrics['term1']:.6f}"
                f"  term2(res²/var): {val_metrics['term2']:.6f}"
                f"  KL: {val_metrics['kl']:.6f}"
            )

        if val_metrics["loss"] < best_val_loss:
            prev_best = best_val_loss
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
                f" — checkpoint saved to {checkpoint_path}"
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
    data.add_argument(
        "--train-npz", required=True,
        help="Training .npz produced by prepare_dataset.py.",
    )
    data.add_argument(
        "--test-npz", default=None,
        help="Optional held-out test .npz; evaluated once after training using the best checkpoint.",
    )
    data.add_argument(
        "--val-split", type=float, default=0.1,
        help="Fraction of training data held out for validation.",
    )

    mdl = parser.add_argument_group("model")
    mdl.add_argument(
        "--input-length", type=int, default=140,
        help="Input sequence length passed to PNASModel.",
    )
    mdl.add_argument(
        "--no-batchnorm", action="store_true",
        help="Disable BatchNorm in ResidualTuner (replace with nn.Identity).",
    )
    mdl.add_argument(
        "--checkpoint", default=None, metavar="PATH",
        help=(
            "Warm-start from this checkpoint. Supports partial checkpoints "
            "(e.g. seq filters only, seq+struct, or full model). "
            "Missing parameters are left at random initialization."
        ),
    )

    opt = parser.add_argument_group("optimization")
    opt.add_argument("--batch-size",    type=int,   default=64)
    opt.add_argument("--epochs",        type=int,   default=100)
    opt.add_argument("--lr",            type=float, default=1e-3)
    opt.add_argument("--weight-decay",  type=float, default=0.0)
    opt.add_argument(
        "--patience", type=int, default=10,
        help="Early stopping: epochs without val loss improvement before halting.",
    )

    unc = parser.add_argument_group("uncertainty")
    unc.add_argument(
        "--uncertainty", action="store_true",
        help=(
            "Train with the variance head using Gaussian NLL in logit space. "
            "Replaces the KL divergence loss."
        ),
    )
    unc.add_argument(
        "--freeze-epochs", type=int, default=5,
        help=(
            "Phase 1 length: number of epochs to train only the variance branch "
            "(variance_bottleneck + variance_tuner) while all other parameters "
            "are frozen. Only used when --uncertainty is set."
        ),
    )
    unc.add_argument(
        "--lr-phase2", type=float, default=None,
        help=(
            "Learning rate for phase 2 (full fine-tune). "
            "Defaults to lr / 10 if not set."
        ),
    )
    unc.add_argument(
        "--lambda", type=float, default=1.0, dest="lambda_",
        help=(
            "Weight on the log_var (uncertainty regularisation) term in the NLL loss. "
            "Increase to encourage wider variance estimates; decrease to prioritise fit. "
            "Only used when --uncertainty is set."
        ),
    )

    run = parser.add_argument_group("runtime")
    run.add_argument(
        "--checkpoint-dir", default="./checkpoints",
        help="Directory for saving best-model checkpoints.",
    )
    run.add_argument(
        "--device", default=None, metavar="DEV",
        help="Torch device string (e.g. 'cpu', 'cuda', 'cuda:1'). Auto-detects if omitted.",
    )
    run.add_argument("--seed", type=int, default=42)

    return parser


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Reproducibility ───────────────────────────────────────────────────────
    logger.info(f"Random seed: {args.seed}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Load training data ────────────────────────────────────────────────────
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

    # ── Dataset and split ─────────────────────────────────────────────────────
    dataset = PSIDataset(x_seq, x_struct, x_wobble, y)
    n_total = len(dataset)
    n_val   = int(args.val_split * n_total)
    n_train = n_total - n_val
    if n_train == 0 or n_val == 0:
        raise ValueError(f"Dataset too small for val_split={args.val_split}.")
    logger.info(
        f"Split (seed={args.seed}) — "
        f"train: {n_train:,} ({1 - args.val_split:.0%}), "
        f"val: {n_val:,} ({args.val_split:.0%})"
    )

    train_dataset, val_dataset = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,  pin_memory=pin,
    )
    val_loader = DataLoader(
        val_dataset,   batch_size=args.batch_size, shuffle=False, pin_memory=pin,
    )
    logger.info(
        f"DataLoaders — train: {len(train_loader)} batches, "
        f"val: {len(val_loader)} batches (batch_size={args.batch_size})"
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    use_batchnorm = not args.no_batchnorm
    logger.info(
        f"Instantiating PNASModel — input_length={args.input_length}, "
        f"use_batchnorm={use_batchnorm}"
    )
    model = PNASModel(input_length=args.input_length, use_batchnorm=use_batchnorm)

    # ── Warm-start from checkpoint ────────────────────────────────────────────
    if args.checkpoint is not None:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        raw        = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state_dict = raw.get("model_state_dict", raw)
        model.load_partial_state_dict(state_dict)
    else:
        logger.info("No checkpoint provided — training from random initialization.")

    # ── Train ─────────────────────────────────────────────────────────────────
    hparams = {
        "device":         device,
        "num_epochs":     args.epochs,
        "patience":       args.patience,
        "checkpoint_dir": args.checkpoint_dir,
        "lr":             args.lr,
        "weight_decay":   args.weight_decay,
        "loss_fn":        kl_divergence_from_logits,
        "uncertainty":    args.uncertainty,
        "freeze_epochs":  args.freeze_epochs,
        "lr_phase2":      args.lr_phase2 if args.lr_phase2 is not None else args.lr * 0.1,
        "lambda_":        args.lambda_,
    }
    results = train(model, train_loader, val_loader, hparams)

    # ── Optional test evaluation ──────────────────────────────────────────────
    if args.test_npz is not None:
        logger.info(f"Loading test data: {args.test_npz}")
        test_npz = np.load(args.test_npz)
        test_dataset = PSIDataset(
            test_npz["seq_oh"],
            test_npz["struct_oh"],
            test_npz["wobbles"],
            test_npz["metadata_PSI"].astype(np.float32),
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=pin,
        )
        logger.info(
            f"  Test — examples: {len(test_dataset):,}, batches: {len(test_loader)}"
        )

        logger.info(f"Reloading best checkpoint for test eval: {results['checkpoint_path']}")
        best_ckpt = torch.load(results["checkpoint_path"], map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        model = model.to(device)

        test_metrics = eval_epoch(model, test_loader, kl_divergence_from_logits, device)
        logger.info(
            f"Test — loss: {test_metrics['loss']:.6f}  rmse: {test_metrics['rmse']:.6f}"
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
