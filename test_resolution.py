"""Check variable-resolution pooling and spatial explanations."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from fusion_model import explain
from pipeline import IMAGE_SIZE, PATCH_COLS, PATCH_ROWS, extract_batch, image_to_tensor


class ResolutionTests(unittest.TestCase):
    def test_high_resolution_pooling_uses_entire_grid(self):
        size = (IMAGE_SIZE[0] * 2, IMAGE_SIZE[1] * 2)
        rows, cols = size[1] // 14, size[0] // 14
        values = torch.arange((rows * cols + 1) * 4, dtype=torch.float32).reshape(1, -1, 4)
        class FakeModel:
            _alliage_image_size = size

            def __call__(self, pixel_values):
                self_shape = tuple(pixel_values.shape)
                if self_shape != (1, 3, size[1], size[0]):
                    raise AssertionError(self_shape)
                return SimpleNamespace(last_hidden_state=values)
        image = Image.new("RGB", (1600, 800), "gray")
        features, patches = extract_batch(FakeModel(), [image], "cpu", "rad-dino")
        self.assertEqual(features.shape, (1, 32))
        self.assertEqual(patches.shape, (1, rows * cols, 4))
        np.testing.assert_allclose(features[0, 4:8], patches[0].mean(axis=0))
        np.testing.assert_allclose(features[0, 8:12], patches[0].reshape(rows, cols, 4)[:rows // 2, :cols // 3].mean(axis=(0, 1)))
        self.assertEqual(image.size, (1600, 800))
        self.assertTrue(torch.isfinite(image_to_tensor(image, "rad-dino", size)).all())

    def test_mixed_resolution_explanations_preserve_total_contribution(self):
        classifier = SimpleNamespace(named_steps={
            "standardscaler": SimpleNamespace(scale_=None),
            "logisticregression": SimpleNamespace(coef_=np.array([[1.] * 64, [0.] * 64])),
        })
        vectors = [np.ones(32), np.ones(32)]
        patches = [np.ones((PATCH_ROWS * PATCH_COLS * factor ** 2, 4)) for factor in (1, 2)]
        heat = explain(classifier, vectors, patches, 0, 1)
        self.assertEqual(heat.shape, (PATCH_ROWS, PATCH_COLS))
        self.assertTrue(np.isfinite(heat).all())
        self.assertAlmostEqual(float(heat.sum()), 2 * 7 * 4 / np.sqrt(32), places=5)


if __name__ == "__main__":
    unittest.main()
