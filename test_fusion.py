"""Check ensemble explanations and feature normalization used by fusion inference."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fusion_model import FusionEnsemble, load_models, normalize_feature_blocks


class FusionTests(unittest.TestCase):
    def test_region_normalization_ignores_positive_block_scaling(self):
        rng = np.random.default_rng(42)
        matrix = rng.normal(size=(3, 40))
        altered = matrix.copy()
        altered[:, :16] = (altered[:, :16].reshape(3, 8, 2) * np.arange(1, 9)[None, :, None]).reshape(3, 16)
        np.testing.assert_allclose(normalize_feature_blocks(matrix, (2, 3)),
                                   normalize_feature_blocks(altered, (2, 3)), atol=1e-12)
        np.testing.assert_array_equal(normalize_feature_blocks(np.zeros((1, 40)), (2, 3)), np.zeros((1, 40)))
        with self.assertRaises(ValueError):
            normalize_feature_blocks(matrix, (2,))

    def test_ensemble_gradient_matches_probability_margin(self):
        rng = np.random.default_rng(42)
        matrix = rng.normal(size=(30, 8))
        labels = np.array(["a", "b", "c"] * 10)
        members = [make_pipeline(StandardScaler(), LogisticRegression(C=c)).fit(matrix, labels)
                   for c in (.1, 1.)]
        ensemble = FusionEnsemble(members)
        np.testing.assert_allclose(ensemble.predict_proba(matrix),
                                   np.mean([member.predict_proba(matrix) for member in members], axis=0))
        feature = matrix[0].copy()
        gradient = ensemble.margin_coefficients(feature, 0, 1)
        differences = []
        for index in range(len(feature)):
            offset = np.zeros_like(feature)
            offset[index] = 1e-5
            plus = ensemble.predict_proba((feature + offset)[None])[0]
            minus = ensemble.predict_proba((feature - offset)[None])[0]
            differences.append(((plus[0] - plus[1]) - (minus[0] - minus[1])) / 2e-5)
        np.testing.assert_allclose(gradient, differences, atol=1e-8)

    def test_loading_preserves_backbone_order_and_resolution(self):
        specs = [{"key": "rad-dino", "image_size": [1036, 504]}, {"key": "dinov2-small"}]
        with patch("fusion_model.load_backbone", side_effect=lambda device, key: SimpleNamespace()):
            models = load_models("cpu", specs)
        self.assertEqual(list(models), ["rad-dino", "dinov2-small"])
        self.assertEqual(models["rad-dino"]._alliage_image_size, (1036, 504))
        self.assertFalse(hasattr(models["dinov2-small"], "_alliage_image_size"))


if __name__ == "__main__":
    unittest.main()
