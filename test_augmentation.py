"""Guard against leakage and normalization/cache regressions in augmentation CV."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from experiments.compare_augmentation import (
    build_classifier,
    cache_key,
    cache_metadata,
    extract_views,
    select_result,
    training_rows,
    view_seed,
)
from fusion_model import unit
from pipeline import BACKBONES, IMAGE_SIZE, augment_image, image_to_tensor


class AugmentationTests(unittest.TestCase):
    def test_training_variants_never_include_validation_sources(self):
        features = np.arange(5 * 4 * 2).reshape(5, 4, 2)
        labels = np.array(["a", "b", "c", "d", "e"])
        x, y = training_rows(features, labels, np.array([0, 2, 4]), 3)
        self.assertEqual(x.shape, (12, 2))
        np.testing.assert_array_equal(y, np.repeat(labels[[0, 2, 4]], 4))
        self.assertFalse(set(x[:, 0]) & set(features[[1, 3]].reshape(-1, 2)[:, 0]))

    def test_scaler_only_fits_training_samples(self):
        model = build_classifier()
        train = np.array([[0., 2.], [1., 4.], [2., 6.], [3., 8.]])
        model.fit(train, ["a", "a", "b", "b"])
        model.predict(np.array([[1000., 2000.]]))
        np.testing.assert_allclose(model.named_steps["standardscaler"].mean_, train.mean(axis=0))

    def test_normalization_and_constant_images(self):
        for key, config in BACKBONES.items():
            image = Image.new("RGB", IMAGE_SIZE, (128, 128, 128))
            tensor = image_to_tensor(image, key).numpy()
            expected = (128 / 255 - np.array(config["mean"])) / np.array(config["std"])
            np.testing.assert_allclose(tensor[:, 100, 100], expected, atol=1e-6)
            self.assertTrue(np.isfinite(tensor).all())
        np.testing.assert_array_equal(unit(np.zeros((2, 4), dtype=np.float32)), np.zeros((2, 4)))
        np.testing.assert_allclose(np.linalg.norm(unit(np.ones((2, 4))), axis=-1), 1)

    def test_cache_invalidates_preprocessing_revision_and_sources(self):
        original = cache_metadata(["a", "b"], "rad-dino", 3)
        for field, value in (("views", 1), ("seed", 43), ("digests", ["b", "a"]),
                             ("implementation", "changed"), ("image_size", (100, 50))):
            changed = copy.deepcopy(original)
            changed[field] = value
            self.assertNotEqual(cache_key(original), cache_key(changed))
        changed = copy.deepcopy(original)
        changed["backbone"]["revision"] = "new"
        self.assertNotEqual(cache_key(original), cache_key(changed))

    def test_same_view_seed_and_pixels_across_backbones(self):
        import tempfile
        from pathlib import Path

        import experiments.compare_augmentation as experiment

        captured = []
        def fake_extract(model, images, device, backbone):
            captured.append((backbone, np.stack([np.asarray(image) for image in images])))
            return np.ones((len(images), 8), dtype=np.float32), None

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            Image.new("RGB", (64, 32), (128, 128, 128)).save(directory / "abc.jpg")
            with patch.object(experiment, "DATA", directory), patch.object(experiment, "IMAGES", directory), \
                    patch.object(experiment, "load_backbone", return_value=None), \
                    patch.object(experiment, "extract_batch", fake_extract):
                for backbone in ("rad-dino", "dinov2-small"):
                    extract_views(["abc"], backbone, 4, 3)
                # Reusing a valid cache must not run extraction again.
                extract_views(["abc"], "rad-dino", 4, 3)
        self.assertEqual(len(captured), 2)
        np.testing.assert_array_equal(captured[0][1], captured[1][1])
        image = Image.new("RGB", (64, 32), (128, 128, 128))
        first = np.asarray(augment_image(image, view_seed("abc", 1)))
        np.testing.assert_array_equal(first, captured[0][1][1])
        self.assertFalse(np.array_equal(first, captured[0][1][2]))

    def test_selection_requires_accuracy_gain_without_f1_loss(self):
        base = {"views": 0, "accuracy": .5, "macro_f1": .45}
        bad = {"views": 1, "accuracy": .6, "macro_f1": .44}
        good = {"views": 3, "accuracy": .55, "macro_f1": .46}
        self.assertEqual(select_result([base, bad, good])["views"], 3)
        self.assertEqual(select_result([base, bad])["views"], 0)
        tied = dict(good, views=1)
        self.assertEqual(select_result([base, good, tied])["views"], 1)


if __name__ == "__main__":
    unittest.main()
