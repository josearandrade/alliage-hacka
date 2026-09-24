"""Focused checks for the data boundary used in the final evaluation."""

import unittest

import numpy as np
from PIL import Image

from pipeline import CLASSES, augment_image, make_splits


class SplitIntegrityTests(unittest.TestCase):
    def test_test_overlap_and_conflicting_training_labels_are_removed(self):
        records = [
            {"hash": f"clean-{label}", "label": label, "set": "dataset"}
            for label in CLASSES
        ]
        records += [
            {"hash": "overlap", "label": "normal", "set": "dataset"},
            {"hash": "overlap", "label": "normal", "set": "teste"},
            {"hash": "contradiction", "label": "normal", "set": "dataset"},
            {"hash": "contradiction", "label": "ampliacao", "set": "dataset"},
        ]
        records += [
            {"hash": f"test-{label}", "label": label, "set": "teste"}
            for label in CLASSES
        ]
        records += [
            {"hash": "test-contradiction", "label": "normal", "set": "teste"},
            {"hash": "test-contradiction", "label": "ampliacao", "set": "teste"},
        ]

        splits = make_splits(records)
        train_hashes = {row["hash"] for row in splits["train_pool"]}
        self.assertNotIn("overlap", train_hashes)
        self.assertNotIn("contradiction", train_hashes)
        self.assertEqual(len(splits["train_pool"]), len(CLASSES))
        self.assertEqual(len(splits["test_raw"]), len(CLASSES) + 3)
        self.assertEqual(len(splits["test_clean"]), len(CLASSES) + 1)
        self.assertEqual(splits["audit"]["cross_split_hashes"], 1)
        self.assertEqual(splits["audit"]["dataset_conflicting_hashes"], 1)
        self.assertEqual(splits["audit"]["test_conflicting_hashes"], 1)

    def test_augmentation_is_deterministic_and_preserves_shape(self):
        image = Image.new("RGB", (64, 32), (128, 128, 128))
        first = np.asarray(augment_image(image, 42))
        second = np.asarray(augment_image(image, 42))
        other = np.asarray(augment_image(image, 43))
        self.assertEqual(first.shape, (32, 64, 3))
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, other))


if __name__ == "__main__":
    unittest.main()
