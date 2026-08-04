"""PyTorch implementation of the PNAS splicing model and related helpers."""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def lanczos_kernel(x, order):
    """Compute Lanczos kernel weights for the given offsets.

    Args:
        x: Scalar or array-like offsets from the interpolation target.
        order: Size of the Lanczos window.

    Returns:
        A scalar or NumPy array of Lanczos kernel weights with the same shape
        as ``x``.
    """
    return np.sinc(x) * np.sinc(x/order) * ((x > -order) * (x < order))

def lanczos_interpolate(arr, positions, order=3):
    """Interpolate a 1D array at arbitrary positions with a Lanczos kernel.

    Args:
        arr: One-dimensional NumPy array to sample from.
        positions: Array-like floating-point sample locations.
        order: Size of the Lanczos window. Defaults to ``3``.

    Returns:
        A NumPy array containing interpolated values for each input position.
    """
    positions = np.asarray(positions)
    if positions.size == 0:
        return np.zeros_like(positions)

    # Evaluate the same 2 * order sample window for every target position at
    # once. Out-of-bounds samples are clipped for indexing and then masked out,
    # preserving the edge behavior of the original per-position loop.
    offsets = np.arange(-order + 1, order + 1)
    sample_indices = np.floor(positions).astype(np.int64)[:, None] + offsets
    valid = (sample_indices >= 0) & (sample_indices < len(arr))
    clipped_indices = np.clip(sample_indices, 0, len(arr) - 1)
    weights = lanczos_kernel(
        positions[:, None] - sample_indices,
        order,
    ) * valid
    return np.sum(arr[clipped_indices] * weights, axis=1)

def lanczos_resampling(arr, new_len, order=3):
    """Resample a 1D array to a new length with Lanczos interpolation.

    Args:
        arr: One-dimensional NumPy array to resample.
        new_len: Number of output samples.
        order: Size of the Lanczos window. Defaults to ``3``.

    Returns:
        A NumPy array of length ``new_len``.
    """
    return lanczos_interpolate(arr, np.linspace(0, len(arr)-1, num=new_len), order)

class SumDiff(nn.Module):
    """Aggregate inclusion and skipping activations into a scalar energy."""

    def __init__(self):
        super(SumDiff, self).__init__()
        self.w = nn.Parameter(torch.randn(1))  # Learnable weight
        self.b = nn.Parameter(torch.zeros(1))   # Learnable bias

    def forward(self, x):
        """Compute the weighted sum-difference score.

        Args:
            x: Tensor of shape ``(batch_size, 2, num_filters, seq_length)``.
                Index ``0`` is treated as inclusion and index ``1`` as skipping.

        Returns:
            Tensor of shape ``(batch_size,)`` containing the scalar energy per
            example.
        """
        # x shape: (batch_size, 2, num_filters, seq_length)
        diff = x[:, 0].sum(dim=(1, 2)) - x[:, 1].sum(dim=(1, 2))
        return self.w * diff + self.b

class ResidualTuner(nn.Module):
    """Residual calibration head used after the energy score.

    This module mirrors the original Keras implementation:

    ``Dense(hidden) -> ReLU -> BatchNorm -> Dense(hidden) -> ReLU
    -> BatchNorm -> Dense(1) -> residual add``.

    The input is expected to have a trailing dimension of size ``1`` so the
    residual addition can be applied directly.
    """
    def __init__(self, hidden_units: int = 100, eps: float = 1e-3, momentum: float = 0.99, use_batchnorm: bool = True):
        """Initialize the tuner network.

        Args:
            hidden_units: Width of the two hidden linear layers.
            eps: Batch normalization epsilon.
            momentum: Keras-style batch normalization momentum. Internally
                converted to the PyTorch convention.
            use_batchnorm: If False, BatchNorm1d layers are replaced with
                nn.Identity. Useful when batch statistics are unreliable
                (e.g. very small batches or fine-tuning runs).
        """
        super().__init__()
        self.hidden_units = hidden_units
        self.use_batchnorm = use_batchnorm

        self.fc1 = nn.Linear(1, hidden_units)          # in_features fixed to 1 to match Dense(?, hidden)
        self.bn1 = (
            nn.BatchNorm1d(hidden_units, eps=eps, momentum=1 - momentum)
            if use_batchnorm else nn.Identity()
        )

        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.bn2 = (
            nn.BatchNorm1d(hidden_units, eps=eps, momentum=1 - momentum)
            if use_batchnorm else nn.Identity()
        )

        self.fc3 = nn.Linear(hidden_units, 1)

        if not use_batchnorm:
            logger.info("ResidualTuner: BatchNorm disabled — bn1 and bn2 replaced with nn.Identity.")

    def forward(self, inp: torch.Tensor):
        """Run the residual calibration network.

        Args:
            inp: Tensor with shape ``(..., 1)``.

        Returns:
            Tensor with the same shape as ``inp`` (logit_mu with residual
            connection).

        Raises:
            ValueError: If the last dimension of ``inp`` is not ``1``.
        """
        if inp.shape[-1] != 1:
            raise ValueError(f"ResidualTuner expects last dim == 1, got {inp.shape[-1]}")

        # Flatten to (N, C) for BatchNorm1d, then restore shape
        orig_shape = inp.shape
        x = inp.reshape(-1, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.bn2(x)

        logit_mu = self.fc3(x).reshape(orig_shape) + inp  # residual connection
        return logit_mu

    @torch.no_grad()
    def load_weights_from_dict(self, weight_dict):
        """Load weights exported from the TensorFlow/Keras tuner.

        Args:
            weight_dict: Mapping containing dense and batch-normalization
                parameters. Expected keys are ``fc1_w``, ``fc1_b``,
                ``bn1_gamma``, ``bn1_beta``, ``bn1_mean``, ``bn1_var``,
                ``fc2_w``, ``fc2_b``, ``bn2_gamma``, ``bn2_beta``,
                ``bn2_mean``, ``bn2_var``, ``fc3_w``, and ``fc3_b``.

        Returns:
            The current module instance.
        """
    
        def _copy(dst, src, transpose=False):
            if transpose:
                src = src.t()
            dst.copy_(src.to(dtype=dst.dtype, device=dst.device))
    
        # ---- Dense 1 ----
        _copy(self.fc1.weight, weight_dict["fc1_w"], transpose=True)
        _copy(self.fc1.bias,   weight_dict["fc1_b"])
        logger.info("ResidualTuner.load_weights_from_dict: loaded fc1 weights.")

        # ---- BN 1 ----
        if self.use_batchnorm:
            _copy(self.bn1.weight,       weight_dict["bn1_gamma"])  # gamma
            _copy(self.bn1.bias,         weight_dict["bn1_beta"])   # beta
            _copy(self.bn1.running_mean, weight_dict["bn1_mean"])
            _copy(self.bn1.running_var,  weight_dict["bn1_var"])
            logger.info("ResidualTuner.load_weights_from_dict: loaded bn1 weights.")
        else:
            logger.warning(
                "ResidualTuner.load_weights_from_dict: BatchNorm disabled — "
                "skipping bn1 weights (bn1_gamma, bn1_beta, bn1_mean, bn1_var)."
            )

        # ---- Dense 2 ----
        _copy(self.fc2.weight, weight_dict["fc2_w"], transpose=True)
        _copy(self.fc2.bias,   weight_dict["fc2_b"])
        logger.info("ResidualTuner.load_weights_from_dict: loaded fc2 weights.")

        # ---- BN 2 ----
        if self.use_batchnorm:
            _copy(self.bn2.weight,       weight_dict["bn2_gamma"])
            _copy(self.bn2.bias,         weight_dict["bn2_beta"])
            _copy(self.bn2.running_mean, weight_dict["bn2_mean"])
            _copy(self.bn2.running_var,  weight_dict["bn2_var"])
            logger.info("ResidualTuner.load_weights_from_dict: loaded bn2 weights.")
        else:
            logger.warning(
                "ResidualTuner.load_weights_from_dict: BatchNorm disabled — "
                "skipping bn2 weights (bn2_gamma, bn2_beta, bn2_mean, bn2_var)."
            )

        # ---- Dense 3 ----
        _copy(self.fc3.weight, weight_dict["fc3_w"], transpose=True)
        _copy(self.fc3.bias,   weight_dict["fc3_b"])
        logger.info("ResidualTuner.load_weights_from_dict: loaded fc3 weights.")

        return self

class VarianceTuner(nn.Module):
    """Interpretable variance head with a linear bottleneck.

    Architecture (takes a scalar input from an external bottleneck)::

        Linear(1, 16) → ReLU
        Linear(16, 16) → ReLU
        Linear(16, 1)
        → Softplus + 1e-6        (var)
        → log(var)               (log_var)

    The 56→1 linear bottleneck lives in PNASModel as ``variance_bottleneck``
    so it is a named, inspectable parameter of the full model.  VarianceTuner
    then maps that single scalar to variance, making the full pipeline an
    interpretable R → R function.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, z: torch.Tensor):
        """Predict log-variance and variance from a scalar bottleneck value.

        Args:
            z: Tensor of shape ``(N, 1)`` — output of the linear bottleneck.

        Returns:
            Tuple ``(log_var, var)`` each of shape ``(N, 1)``.
        """
        x = F.relu(self.fc1(z))         # (N, 16)
        x = F.relu(self.fc2(x))         # (N, 16)
        x = self.fc3(x)                 # (N, 1)
        var     = F.softplus(x) + 1e-6
        log_var = torch.log(var)
        return log_var, var


class PNASModel(nn.Module):
    """Inference model for exon inclusion prediction from sequence and structure."""

    def __init__(self, input_length=140, seq_in_channels=4, struct_in_channels=3, wobble_in_channels=1, use_batchnorm=True):
        """Initialize the model architecture.

        Args:
            input_length: Total length of the input window, including flanking
                context. Defaults to ``140`` (40 nt left + 70 nt exon + 30 nt right).
            use_batchnorm: Passed through to ResidualTuner. If False, the two
                BatchNorm1d layers in the tuner are replaced with nn.Identity.
        """
        super(PNASModel, self).__init__()
        self.input_length = input_length
        # Training stage (1/2/3). Not an nn.Parameter — set externally by train_staged().
        #   1: seq filters only, SumDiff+sigmoid (SimplifiedTuner)
        #   2: seq+struct filters, SumDiff+sigmoid (SimplifiedTuner)
        #   3: seq+struct filters, full ResidualTuner (normal inference)
        self.stage = 3

        # In channels for sequence, structure, and wobble inputs.
        self.seq_in_channels = seq_in_channels
        self.struct_in_channels = struct_in_channels
        self.wobble_in_channels = wobble_in_channels
        self.total_struct_channels = self.seq_in_channels + self.struct_in_channels + self.wobble_in_channels
    
        # Fixed hyperparameters from original PNAS model
        self.seq_kernel_size = 6
        self.struct_kernel_size = 30
        self.num_seq_filters = 20
        self.num_struct_filters = 8
        
        ### Sequence layers ###
        # (valid padding) #
        self.conv_skip = nn.Conv1d(in_channels=self.seq_in_channels, out_channels=self.num_seq_filters, kernel_size=self.seq_kernel_size, padding=0)
        self.conv_incl = nn.Conv1d(in_channels=self.seq_in_channels, out_channels=self.num_seq_filters, kernel_size=self.seq_kernel_size, padding=0)

        # Position bias layers
        conv_out_shape = input_length - self.seq_kernel_size + 1
        self.position_bias_skip = nn.Parameter(torch.zeros(self.num_seq_filters, conv_out_shape))
        self.position_bias_incl = nn.Parameter(torch.zeros(self.num_seq_filters, conv_out_shape))

        ### Structure layers ###
        # (same padding) #
        self.conv_struct_skip = nn.Conv1d(in_channels=self.total_struct_channels, out_channels=self.num_struct_filters, kernel_size=self.struct_kernel_size, padding='same')
        self.conv_struct_incl = nn.Conv1d(in_channels=self.total_struct_channels, out_channels=self.num_struct_filters, kernel_size=self.struct_kernel_size, padding='same')
        self.position_bias_skip_struct = nn.Parameter(torch.zeros(self.num_struct_filters, input_length))
        self.position_bias_incl_struct = nn.Parameter(torch.zeros(self.num_struct_filters, input_length))

        ### Aggregation ###
        self.energy_seq_struct = SumDiff()

        ### Activation ###
        self.energy_activation_incl = nn.Softplus()
        self.energy_activation_skip = nn.Softplus()

        ### Tuner ###
        self.tuner = ResidualTuner(hidden_units=4, use_batchnorm=use_batchnorm)
        self.variance_bottleneck = nn.Linear(56, 1)  # interpretable linear projection, no activation
        self.variance_tuner = VarianceTuner()
        self.output_activation = nn.Sigmoid()

        logger.info(
            f"PNASModel initialized — input_length={input_length}, "
            f"use_batchnorm={use_batchnorm}, "
            f"total parameters: {sum(p.numel() for p in self.parameters()):,}"
        )

    @torch.no_grad()
    def load_weights_from_dict(self, parameter_dict):
        """Load a parameter dictionary exported outside of PyTorch.

        Args:
            parameter_dict: Mapping containing convolution, bias, aggregation,
                and tuner parameters. The nested ``"tuner"`` key is forwarded to
                :meth:`ResidualTuner.load_weights_from_dict`.

        Returns:
            The current model instance.
        """
        def _to_like(t, ref):
            return t.to(dtype=ref.dtype, device=ref.device)
    
        def _copy_param(dst, src):
            dst.copy_(_to_like(src, dst))
    
        def _load_conv1d(conv: nn.Conv1d, w_key: str, b_key: str):
            w = parameter_dict[w_key]
            b = parameter_dict[b_key]
            _copy_param(conv.weight, w)
            _copy_param(conv.bias, b)
    
        # -------------------------
        # Sequence conv + pos bias
        # -------------------------
        _load_conv1d(self.conv_incl, "conv_incl_w", "conv_incl_b")
        _load_conv1d(self.conv_skip, "conv_skip_w", "conv_skip_b")
        _copy_param(self.position_bias_incl, parameter_dict["position_bias_incl"])
        _copy_param(self.position_bias_skip, parameter_dict["position_bias_skip"])
    
        # -------------------------
        # Structure conv + pos bias
        # -------------------------
        _load_conv1d(self.conv_struct_incl, "conv_struct_incl_w", "conv_struct_incl_b")
        _load_conv1d(self.conv_struct_skip, "conv_struct_skip_w", "conv_struct_skip_b")
        _copy_param(self.position_bias_incl_struct, parameter_dict["position_bias_incl_struct"])
        _copy_param(self.position_bias_skip_struct, parameter_dict["position_bias_skip_struct"])
    
        # -------------------------
        # SumDiff (energy_seq_struct)
        # -------------------------
        _copy_param(self.energy_seq_struct.w, parameter_dict["energy_seq_struct_w"])
        _copy_param(self.energy_seq_struct.b, parameter_dict["energy_seq_struct_b"])

        tuner_params = parameter_dict['tuner']
        self.tuner.load_weights_from_dict(tuner_params)
    
        return self

    def forward(self, x_seq, x_struct, x_wobble, return_logits=False, return_uncertainty=False):
        """Compute exon inclusion probabilities.

        Args:
            x_seq: Sequence tensor of shape ``(batch_size, 4, input_length)``.
            x_struct: Structure tensor of shape ``(batch_size, 3, input_length)``.
            x_wobble: Wobble tensor of shape ``(batch_size, 1, input_length)``.
            return_logits: If True, return raw logits instead of sigmoid PSI.
            return_uncertainty: If True, also return ``(log_var, var)`` from
                ``VarianceTuner`` alongside the main prediction.

        Returns:
            If both flags are False: sigmoid PSI tensor of shape ``(batch_size,)``.
            If ``return_logits`` is True: logit_mu of shape ``(batch_size,)``.
            If ``return_uncertainty`` is True: tuple ``(prediction, log_var, var)``
            where prediction is logits or PSI depending on ``return_logits``,
            and ``log_var``/``var`` each have shape ``(batch_size,)``.
        """
        # ── Sequence activations (all stages) ────────────────────────────────
        conv_skip_out = self.conv_skip(x_seq) + self.position_bias_skip.unsqueeze(0)
        conv_incl_out = self.conv_incl(x_seq) + self.position_bias_incl.unsqueeze(0)

        if self.stage == 1:
            # Seq-only: bypass struct conv entirely so no gradient flows through it.
            activations_skip = self.energy_activation_skip(conv_skip_out)  # (B, F_seq, L-5)
            activations_incl = self.energy_activation_incl(conv_incl_out)
        else:
            # ── Structure activations (stages 2 and 3) ───────────────────────
            struct_input = torch.cat([x_seq, x_struct, x_wobble], dim=1)
            conv_struct_skip_out = self.conv_struct_skip(struct_input) + self.position_bias_skip_struct.unsqueeze(0)
            conv_struct_incl_out = self.conv_struct_incl(struct_input) + self.position_bias_incl_struct.unsqueeze(0)
            # Crop to match seq length (seq kernel 6, valid padding → L-5)
            conv_struct_skip_out = conv_struct_skip_out[:, :, 2:-3]
            conv_struct_incl_out = conv_struct_incl_out[:, :, 2:-3]
            activations_skip = self.energy_activation_skip(torch.cat([conv_skip_out, conv_struct_skip_out], dim=1))
            activations_incl = self.energy_activation_incl(torch.cat([conv_incl_out, conv_struct_incl_out], dim=1))

        # ── Variance branch (stage 3 only, when requested) ───────────────────
        if return_uncertainty:
            h = torch.cat([activations_incl, activations_skip], dim=1).mean(dim=2)
            z = self.variance_bottleneck(h)
            log_var, var = self.variance_tuner(z)

        # ── SumDiff aggregation ───────────────────────────────────────────────
        energy_in  = torch.stack([activations_incl, activations_skip], dim=1)
        energy_out = self.energy_seq_struct(energy_in)  # (B,) — w*diff + b

        # ── Output: simplified tuner (stages 1/2) or full ResidualTuner (stage 3)
        if self.stage in (1, 2):
            # SimplifiedTuner = SumDiff + sigmoid; energy_out is already w*diff+b
            pred = energy_out if return_logits else torch.sigmoid(energy_out)
        else:
            tuner_out = self.tuner(energy_out.unsqueeze(-1))  # (B, 1)
            pred = tuner_out.squeeze() if return_logits else self.output_activation(tuner_out).squeeze()

        if return_uncertainty:
            return pred, log_var.squeeze(), var.squeeze()
        return pred

    @torch.no_grad()
    def shift_output_bias_(self, delta: float | torch.Tensor):
        """Add a constant offset to every output logit.

        The final dense layer in :class:`ResidualTuner` is followed by a
        residual addition, so changing ``tuner.fc3.bias`` by ``delta`` changes
        every final logit by exactly the same amount. No other learned
        parameter or relative logit difference is affected.

        Args:
            delta: Scalar logit offset to add to ``tuner.fc3.bias``.

        Returns:
            The current model instance.

        Raises:
            ValueError: If ``delta`` is not scalar or is not finite.
        """
        delta_tensor = torch.as_tensor(
            delta,
            dtype=self.tuner.fc3.bias.dtype,
            device=self.tuner.fc3.bias.device,
        )
        if delta_tensor.numel() != 1:
            raise ValueError(
                f"Output-bias delta must be scalar, got shape {tuple(delta_tensor.shape)}."
            )
        if not torch.isfinite(delta_tensor).item():
            raise ValueError("Output-bias delta must be finite.")

        self.tuner.fc3.bias.add_(delta_tensor.reshape_as(self.tuner.fc3.bias))
        return self

    def compute_sequence_activations(self, x_seq, agg='mean'):
        """Summarize sequence filter activations for inclusion and skipping.

        Args:
            x_seq: Sequence tensor of shape ``(batch_size, 4, input_length)``.
            agg: Aggregation to apply over the sequence axis. Supported values
                are ``"mean"`` and ``"sum"``.

        Returns:
            A tuple ``(a_incl, a_skip)`` where each tensor has shape
            ``(batch_size, 20)`` after aggregation.

        Raises:
            ValueError: If ``agg`` is not supported.
        """
        conv_skip_out = self.conv_skip(x_seq) + self.position_bias_skip.unsqueeze(0)  # Add position bias
        conv_incl_out = self.conv_incl(x_seq) + self.position_bias_incl.unsqueeze(0)
        a_skip, a_incl = F.softplus(conv_skip_out), F.softplus(conv_incl_out)


        if agg == 'mean':
            a_incl = torch.mean(a_incl, dim=2)
            a_skip = torch.mean(a_skip, dim=2)
        elif agg == 'sum':
            a_incl = torch.sum(a_incl, dim=2)
            a_skip = torch.sum(a_skip, dim=2)
        else:
            raise ValueError(f"Unknown aggregation: {agg}")

        return a_incl, a_skip

    def compute_structure_activations(
        self,
        x_seq,
        x_struct,
        x_wobble,
        agg='mean',
    ):
        """Summarize structure filter activations for inclusion and skipping.

        This applies the same structure-input concatenation, convolutions,
        position biases, crop, and softplus activation used by :meth:`forward`.

        Args:
            x_seq: Sequence tensor of shape ``(batch_size, 4, input_length)``.
            x_struct: Structure tensor of shape
                ``(batch_size, 3, input_length)``.
            x_wobble: Wobble tensor of shape
                ``(batch_size, 1, input_length)``.
            agg: Aggregation to apply over the sequence axis. Supported values
                are ``"mean"`` and ``"sum"``.

        Returns:
            A tuple ``(a_incl, a_skip)`` where each tensor has shape
            ``(batch_size, 8)`` after aggregation.

        Raises:
            ValueError: If ``agg`` is not supported.
        """
        struct_input = torch.cat([x_seq, x_struct, x_wobble], dim=1)
        conv_skip_out = (
            self.conv_struct_skip(struct_input)
            + self.position_bias_skip_struct.unsqueeze(0)
        )
        conv_incl_out = (
            self.conv_struct_incl(struct_input)
            + self.position_bias_incl_struct.unsqueeze(0)
        )

        # Match the sequence convolution's valid output length, as in forward.
        conv_skip_out = conv_skip_out[:, :, 2:-3]
        conv_incl_out = conv_incl_out[:, :, 2:-3]
        a_skip = self.energy_activation_skip(conv_skip_out)
        a_incl = self.energy_activation_incl(conv_incl_out)

        if agg == 'mean':
            a_incl = torch.mean(a_incl, dim=2)
            a_skip = torch.mean(a_skip, dim=2)
        elif agg == 'sum':
            a_incl = torch.sum(a_incl, dim=2)
            a_skip = torch.sum(a_skip, dim=2)
        else:
            raise ValueError(f"Unknown aggregation: {agg}")

        return a_incl, a_skip

    def compute_sr_balance(self, x_seq, agg='mean'):
        """Compute the net inclusion-minus-skipping sequence score.

        Args:
            x_seq: Sequence tensor of shape ``(batch_size, 4, input_length)``.
            agg: Aggregation mode passed to
                :meth:`compute_sequence_activations`.

        Returns:
            Tensor of shape ``(batch_size,)`` containing the summed balance per
            example.
        """
        a_incl, a_skip = self.compute_sequence_activations(x_seq, agg)
        return a_incl.sum(dim=1) - a_skip.sum(dim=1)

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load a PyTorch state dict, resampling position biases when needed.

        This override allows checkpoints trained with a different input length
        to be adapted by Lanczos-resampling the position-bias tensors.

        Args:
            state_dict: Standard PyTorch state dictionary.
            strict: Passed through to ``nn.Module.load_state_dict``.

        Returns:
            The return value of ``nn.Module.load_state_dict``.
        """
        sd = dict(state_dict)  # shallow copy

        F = 30  # shorter flank length (right=30, left=40) — used as conservative padding anchor
        margin = 5

        pad_seq = min(F + margin, (self.input_length - self.seq_kernel_size + 1)//2)
        pad_struct = min(F + (self.struct_kernel_size - 1)//2 + margin, self.input_length//2)
        target_seq_len = self.input_length - self.seq_kernel_size + 1

        # --- sequence pos bias: shape (num_seq_filters, input_length - seq_kernel + 1)
        if "position_bias_skip" in sd:
            src_len = sd["position_bias_skip"].shape[-1]
            if src_len != target_seq_len:
                logger.info(
                    f"Resampling position_bias_skip: {src_len} -> {target_seq_len} "
                    f"(padding={pad_seq})"
                )
            sd["position_bias_skip"] = self._resample_position_bias(
                sd["position_bias_skip"],
                out_len=target_seq_len,
                padding=pad_seq,
            )
        if "position_bias_incl" in sd:
            src_len = sd["position_bias_incl"].shape[-1]
            if src_len != target_seq_len:
                logger.info(
                    f"Resampling position_bias_incl: {src_len} -> {target_seq_len} "
                    f"(padding={pad_seq})"
                )
            sd["position_bias_incl"] = self._resample_position_bias(
                sd["position_bias_incl"],
                out_len=target_seq_len,
                padding=pad_seq,
            )

        # --- structure pos bias: shape (num_struct_filters, input_length)  (NO kernel_size term)
        if "position_bias_skip_struct" in sd:
            src_len = sd["position_bias_skip_struct"].shape[-1]
            if src_len != self.input_length:
                logger.info(
                    f"Resampling position_bias_skip_struct: {src_len} -> {self.input_length} "
                    f"(padding={pad_struct})"
                )
            sd["position_bias_skip_struct"] = self._resample_position_bias(
                sd["position_bias_skip_struct"],
                out_len=self.input_length,
                padding=pad_struct,
            )
        if "position_bias_incl_struct" in sd:
            src_len = sd["position_bias_incl_struct"].shape[-1]
            if src_len != self.input_length:
                logger.info(
                    f"Resampling position_bias_incl_struct: {src_len} -> {self.input_length} "
                    f"(padding={pad_struct})"
                )
            sd["position_bias_incl_struct"] = self._resample_position_bias(
                sd["position_bias_incl_struct"],
                out_len=self.input_length,
                padding=pad_struct,
            )

        return super().load_state_dict(sd, strict=strict)

    def load_partial_state_dict(self, state_dict):
        """Load a (possibly partial) state dict with strict=False.

        Keys present in ``state_dict`` are loaded into the model (with
        position-bias resampling applied as needed). Keys absent from
        ``state_dict`` are left at their current values — randomly initialized
        if the model is fresh.

        Typical use cases for staged training:

        * Seq-only warm-start: checkpoint contains only conv_skip/incl and
          their position biases; SumDiff, tuner, and struct layers stay random.
        * Seq + struct warm-start: as above but also includes struct convs and
          struct position biases.
        * Full warm-start: all parameters present; equivalent to a strict load.

        Args:
            state_dict: Mapping of parameter names to tensors. Checkpoints
                saved by this training script nest weights under
                ``"model_state_dict"``; extract that key before calling here.

        Returns:
            The ``NamedTuple`` returned by ``nn.Module.load_state_dict``
            (contains ``missing_keys`` and ``unexpected_keys``).
        """
        model_keys = set(self.state_dict().keys())
        ckpt_keys  = set(state_dict.keys())

        will_load       = sorted(model_keys & ckpt_keys)
        random_init     = sorted(model_keys - ckpt_keys)
        unexpected_ckpt = sorted(ckpt_keys  - model_keys)

        logger.info("=== load_partial_state_dict ===")
        logger.info(f"  Model parameters:      {len(model_keys)}")
        logger.info(f"  Checkpoint parameters: {len(ckpt_keys)}")
        logger.info(f"  Will be loaded ({len(will_load)}):")
        for k in will_load:
            logger.info(f"    [LOAD]  {k}")
        if random_init:
            logger.info(f"  Kept at current/random init ({len(random_init)}):")
            for k in random_init:
                logger.info(f"    [INIT]  {k}")
        if unexpected_ckpt:
            logger.warning(f"  Unexpected checkpoint keys — will be ignored ({len(unexpected_ckpt)}):")
            for k in unexpected_ckpt:
                logger.warning(f"    [SKIP]  {k}")

        result = self.load_state_dict(state_dict, strict=False)
        logger.info(
            f"  Load result — missing: {len(result.missing_keys)}, "
            f"unexpected: {len(result.unexpected_keys)}"
        )
        logger.info("=== load_partial_state_dict complete ===")
        return result

    def _resample_position_bias(self, orig_weight: torch.Tensor, out_len: int, padding: int):
        """Resample a position-bias tensor while preserving edge padding.

        Args:
            orig_weight: Tensor of shape ``(channels, old_length)``.
            out_len: Target output length.
            padding: Number of values to preserve at both ends before
                resampling the middle segment.

        Returns:
            Tensor of shape ``(channels, out_len)`` on the same device and with
            the same dtype as ``orig_weight``.
        """
        # Ensure CPU numpy conversion is safe
        w = orig_weight.detach().cpu().numpy()  # (C, L_old)
    
        def resample_one_channel(x):
            # x: (L_old,)
            left = x[:padding]
            mid  = x[padding:-padding]
            right = x[-padding:]
    
            # lanczos_resampling(mid, new_mid_len) must exist and return 1D array
            new_mid_len = out_len - 2 * padding
            new_mid = lanczos_resampling(mid, new_mid_len)
    
            return np.concatenate([left, new_mid, right], axis=0)
    
        new_w = np.apply_along_axis(resample_one_channel, 1, w)  # (C, out_len)
        return torch.from_numpy(new_w).to(dtype=orig_weight.dtype, device=orig_weight.device)
