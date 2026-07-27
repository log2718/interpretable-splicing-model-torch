"""Tests for the dataframe prediction API."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from model import PNASModel
from predict import predict


def _zero_model(input_length: int = 40) -> PNASModel:
    model = PNASModel(input_length=input_length, use_batchnorm=False)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    model.eval()
    return model


def _mock_features(n_examples: int, input_length: int = 40):
    return (
        np.zeros((n_examples, 4, input_length), dtype=np.float32),
        np.zeros((n_examples, 3, input_length), dtype=np.float32),
        np.zeros((n_examples, 1, input_length), dtype=np.float32),
    )


class DataFramePredictTests(unittest.TestCase):
    def test_requires_flanked_sequence(self):
        with self.assertRaisesRegex(ValueError, "flanked_sequence"):
            predict(pd.DataFrame({"sequence": ["A" * 40]}), model=_zero_model())

    def test_calibration_requires_psi(self):
        df = pd.DataFrame({"flanked_sequence": ["A" * 40]})

        with self.assertRaisesRegex(ValueError, "PSI"):
            predict(df, calibrate=True, model=_zero_model())

    @patch("predict.create_input_data")
    def test_predicts_and_preserves_input_dataframe(self, create_input_data_mock):
        create_input_data_mock.return_value = _mock_features(2)
        df = pd.DataFrame(
            {
                "flanked_sequence": ["A" * 40, "C" * 40],
                "metadata": ["first", "second"],
            },
            index=[4, 9],
        )

        result = predict(df, model=_zero_model(), device="cpu", batch_size=1)

        self.assertNotIn("predicted_PSI", df.columns)
        self.assertEqual(result.index.tolist(), [4, 9])
        self.assertEqual(result["metadata"].tolist(), ["first", "second"])
        np.testing.assert_allclose(result["predicted_PSI"], [0.5, 0.5])
        create_input_data_mock.assert_called_once()
        self.assertFalse(
            create_input_data_mock.call_args.kwargs["add_flanks"]
        )

    @patch("predict.create_input_data")
    def test_calibrates_entire_dataframe_and_restores_model_bias(
        self,
        create_input_data_mock,
    ):
        create_input_data_mock.return_value = _mock_features(3)
        df = pd.DataFrame(
            {
                "flanked_sequence": ["A" * 40, "C" * 40, "G" * 40],
                "PSI": [0.1, 0.2, 0.3],
            }
        )
        model = _zero_model()
        original_bias = model.tuner.fc3.bias.detach().clone()

        result = predict(
            df,
            calibrate=True,
            model=model,
            device="cpu",
            batch_size=2,
        )

        np.testing.assert_allclose(
            result["predicted_PSI_uncalibrated"],
            [0.5, 0.5, 0.5],
        )
        np.testing.assert_allclose(
            result["predicted_PSI"],
            [0.2, 0.2, 0.2],
            atol=1e-6,
        )
        self.assertEqual(
            result.attrs["bias_calibration"]["n_examples"],
            len(df),
        )
        torch.testing.assert_close(model.tuner.fc3.bias, original_bias)


if __name__ == "__main__":
    unittest.main()
