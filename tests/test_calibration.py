"""Tests for output-bias calibration."""

from __future__ import annotations

import math
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from calibration import collect_logits, fit_logit_bias, tune_output_bias_
from model import PNASModel


def _zero_model(input_length: int = 40) -> PNASModel:
    model = PNASModel(input_length=input_length, use_batchnorm=False)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return model


def _inputs(n_examples: int, input_length: int = 40):
    return (
        torch.zeros(n_examples, 4, input_length),
        torch.zeros(n_examples, 3, input_length),
        torch.zeros(n_examples, 1, input_length),
    )


class FitLogitBiasTests(unittest.TestCase):
    def test_minimizes_probability_rmse(self):
        logits = torch.tensor([-2.0, -0.5, 0.25, 1.5])
        targets = torch.tensor([0.1, 0.4, 0.8, 0.9])

        delta, at_bound = fit_logit_bias(logits, targets)

        self.assertFalse(at_bound)
        fitted_rmse = torch.sqrt(
            torch.mean((torch.sigmoid(logits + delta) - targets) ** 2)
        )
        left_rmse = torch.sqrt(
            torch.mean((torch.sigmoid(logits + delta - 1e-3) - targets) ** 2)
        )
        right_rmse = torch.sqrt(
            torch.mean((torch.sigmoid(logits + delta + 1e-3) - targets) ** 2)
        )
        self.assertLessEqual(fitted_rmse, left_rmse)
        self.assertLessEqual(fitted_rmse, right_rmse)

    def test_supports_sample_weights(self):
        logits = torch.tensor([-1.0, 0.0, 2.0])
        targets = torch.tensor([0.0, 0.25, 1.0])
        weights = torch.tensor([1.0, 2.0, 5.0])

        delta, at_bound = fit_logit_bias(
            logits,
            targets,
            sample_weight=weights,
        )

        self.assertFalse(at_bound)
        probabilities = torch.sigmoid(logits + delta)
        fitted_rmse = torch.sqrt(
            (((probabilities - targets) ** 2) * weights).sum() / weights.sum()
        )
        unweighted_delta, _ = fit_logit_bias(logits, targets)
        unweighted_probabilities = torch.sigmoid(logits + unweighted_delta)
        unweighted_fit_rmse = torch.sqrt(
            (
                ((unweighted_probabilities - targets) ** 2) * weights
            ).sum()
            / weights.sum()
        )
        self.assertLess(fitted_rmse, unweighted_fit_rmse)

    def test_degenerate_targets_report_bound(self):
        delta, at_bound = fit_logit_bias(
            torch.zeros(3),
            torch.zeros(3),
            max_abs_delta=5.0,
        )

        self.assertTrue(at_bound)
        self.assertAlmostEqual(delta, -5.0, delta=1e-6)

    def test_rejects_targets_outside_probability_range(self):
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            fit_logit_bias(torch.zeros(2), torch.tensor([0.5, 1.1]))


class ModelCalibrationTests(unittest.TestCase):
    def test_forward_preserves_singleton_batch_dimension(self):
        model = _zero_model()
        seq, struct, wobble = _inputs(1)

        logits = model(seq, struct, wobble, return_logits=True)
        predictions = model(seq, struct, wobble)

        self.assertEqual(logits.shape, (1,))
        self.assertEqual(predictions.shape, (1,))

    def test_only_final_bias_changes(self):
        model = _zero_model()
        model.train()
        seq, struct, wobble = _inputs(5)
        targets = torch.tensor([0.1, 0.2, 0.4, 0.7, 0.8])
        loader = DataLoader(
            TensorDataset(seq, struct, wobble, targets),
            batch_size=2,
            shuffle=False,
        )
        original_state = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }

        result = tune_output_bias_(model, loader)

        self.assertTrue(model.training)
        self.assertEqual(result.n_examples, 5)
        self.assertLessEqual(result.rmse_after, result.rmse_before)
        self.assertAlmostEqual(
            result.calibrated_prediction_mean,
            result.target_mean,
            places=7,
        )
        for name, original_tensor in original_state.items():
            current_tensor = model.state_dict()[name]
            if name == "tuner.fc3.bias":
                self.assertFalse(torch.equal(current_tensor, original_tensor))
            else:
                self.assertTrue(
                    torch.equal(current_tensor, original_tensor),
                    msg=f"Unexpected parameter or buffer change: {name}",
                )

    def test_calibrated_logits_are_raw_logits_plus_delta(self):
        model = _zero_model()
        seq, struct, wobble = _inputs(3)
        feature_loader = DataLoader(
            TensorDataset(seq, struct, wobble),
            batch_size=2,
            shuffle=False,
        )
        targets = torch.tensor([0.2, 0.4, 0.9])
        calibration_loader = DataLoader(
            TensorDataset(seq, struct, wobble, targets),
            batch_size=2,
            shuffle=False,
        )

        raw_logits = collect_logits(model, feature_loader)
        result = tune_output_bias_(model, calibration_loader)
        calibrated_logits = collect_logits(model, feature_loader)

        torch.testing.assert_close(
            calibrated_logits,
            raw_logits + result.bias_delta,
        )
        expected_delta = math.log(targets.mean().item() / (1 - targets.mean().item()))
        self.assertAlmostEqual(result.bias_delta, expected_delta, places=6)


if __name__ == "__main__":
    unittest.main()
