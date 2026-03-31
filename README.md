# torch-pnas-splicing-model

PyTorch implementation of a splicing model that combines sequence features, structure features, and wobble features to predict exon inclusion.

## Files

- `model.py`: model definition, weight-loading helpers, and position-bias resampling utilities.
- `utils.py`: sequence preprocessing helpers for adding flanks and one-hot encoding nucleotide strings.
- `model_weights.pt`: serialized model weights.

## Requirements

- Python 3.9+
- PyTorch
- NumPy

Install the Python dependencies with your preferred environment manager, for example:

```bash
pip install torch numpy
```

## Input requirements

Flanking is always required. The helper `utils.add_flanking()` prepends and appends the fixed 10 nt context expected by the model, so a 70 nt core exon becomes the default 90 nt input window.

`PNASModel.forward()` expects three tensors with a shared sequence length:

- `x_seq`: shape `(batch_size, 4, input_length)` for one-hot nucleotide channels `A/C/G/T`
- `x_struct`: shape `(batch_size, 3, input_length)` for structure-derived channels, if available
- `x_wobble`: shape `(batch_size, 1, input_length)` for wobble-derived channels, if available

The forward pass returns sigmoid-transformed predictions. Note that the current implementation uses `squeeze()`, so a batch of size `1` is returned as a scalar.

## Full prediction with sequence, structure, and wobble

```python
import torch

from model import PNASModel
from utils import add_flanking, one_hot_batch

core_exons = ["ATGCGT" * 10]
seqs = add_flanking(core_exons)

# one_hot_batch returns (N, L, 4); transpose to channel-first for PyTorch.
x_seq = torch.tensor(one_hot_batch(seqs), dtype=torch.float32).permute(0, 2, 1)
x_struct = torch.zeros((x_seq.shape[0], 2, x_seq.shape[-1]), dtype=torch.float32)
x_wobble = torch.zeros((x_seq.shape[0], 2, x_seq.shape[-1]), dtype=torch.float32)

model = PNASModel(input_length=x_seq.shape[-1])
state_dict = torch.load("model_weights.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

with torch.no_grad():
    prediction = model(x_seq, x_struct, x_wobble)
```

## Sequence-only analysis

If structure and wobble features are not available, you cannot call `forward()`, but you can still inspect the sequence branch with `compute_sequence_activations()` and `compute_sr_balance()`.

```python
import torch

from model import PNASModel
from utils import add_flanking, one_hot_batch

core_exons = ["ATGCGT" * 10]
seqs = add_flanking(core_exons)
x_seq = torch.tensor(one_hot_batch(seqs), dtype=torch.float32).permute(0, 2, 1)

model = PNASModel(input_length=x_seq.shape[-1])
state_dict = torch.load("model_weights.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

with torch.no_grad():
    a_incl, a_skip = model.compute_sequence_activations(x_seq, agg="mean")
    sr_balance = model.compute_sr_balance(x_seq, agg="mean")
```

## Notes

- `load_state_dict()` in `PNASModel` is overridden to resample position-bias tensors when checkpoint and runtime input lengths differ.
- `load_weights_from_dict()` is available for loading weights converted from an external TensorFlow/Keras export format.
- Sequence-only helpers use the sequence convolution branch only; they do not reproduce the full model output.
