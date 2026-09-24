"""Compare training-only acquisition augmentations with fixed fusion settings."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fusion_model import BACKBONE_KEYS, load_models, predict_fusion_image, report, unit
from pipeline import (
    ARTIFACTS,
    BACKBONES,
    CLASSES,
    DATA,
    IMAGE_SIZE,
    IMAGES,
    SEED,
    augment_image,
    extract_batch,
    image_to_tensor,
    load_backbone,
    load_json,
    save_json,
)

OUTPUT = ARTIFACTS / "augmentation"
VIEW_COUNTS = (0, 1, 3)


def view_seed(digest: str, view: int) -> int:
    payload = f"{SEED}:{digest}:{view}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def cache_metadata(digests: list[str], backbone: str, views: int) -> dict:
    return {
        "version": 1, "digests": digests, "backbone": BACKBONES[backbone],
        "image_size": IMAGE_SIZE, "views": views, "seed": SEED,
        "implementation": "\n".join(inspect.getsource(fn) for fn in (
            augment_image, image_to_tensor, extract_batch, view_seed,
        )),
    }


def cache_key(metadata: dict) -> str:
    return hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()


def extract_views(digests: list[str], backbone: str, batch_size: int, views: int) -> np.ndarray:
    """Return [source, original + views, feature]; both backbones share view seeds."""
    metadata = cache_metadata(digests, backbone, views)
    directory = DATA / "augmentation_cache"
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory / f"{backbone}_{cache_key(metadata)}.npz"
    if cache.exists():
        with np.load(cache) as saved:
            matrix = saved["features"]
            if (matrix.shape[:2] == (len(digests), views + 1)
                    and saved["digests"].tolist() == digests and np.isfinite(matrix).all()):
                print(f"Cache: {backbone}, {len(digests)} originals, {views} views", flush=True)
                return matrix
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_backbone(device, backbone)
    items = [(digest, view) for digest in digests for view in range(views + 1)]
    batches = []
    for start in range(0, len(items), batch_size):
        images = []
        for digest, view in items[start:start + batch_size]:
            with Image.open(IMAGES / f"{digest}.jpg") as source:
                images.append(source.copy() if view == 0 else augment_image(source, view_seed(digest, view)))
        vectors, _ = extract_batch(model, images, device, backbone)
        if not np.isfinite(vectors).all():
            raise ValueError(f"Non-finite features: {backbone}, batch {start}")
        batches.append(vectors)
        if start % (batch_size * 50) == 0:
            print(f"{backbone}: {min(start + batch_size, len(items))}/{len(items)}", flush=True)
    matrix = np.concatenate(batches).reshape(len(digests), views + 1, -1)
    np.savez_compressed(cache, features=matrix, digests=np.array(digests))
    save_json(cache.with_suffix(".json"), metadata)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return matrix


def fused_views(digests: list[str], batch_size: int, views: int) -> np.ndarray:
    return np.concatenate([
        unit(extract_views(digests, key, batch_size, views)) for key in BACKBONE_KEYS
    ], axis=-1)


def training_rows(features, labels, indices, views):
    return features[indices, :views + 1].reshape(-1, features.shape[-1]), np.repeat(labels[indices], views + 1)


def build_classifier():
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=0.1, class_weight="balanced", max_iter=1500,
    ))


def metrics(actual, predicted) -> dict:
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=CLASSES, average="macro", zero_division=0)),
        "classification_report": classification_report(actual, predicted, labels=CLASSES, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=CLASSES).tolist(),
    }


def select_result(results: list[dict]) -> dict:
    baseline = next(row for row in results if row["views"] == 0)
    eligible = [row for row in results if row["accuracy"] > baseline["accuracy"]
                and row["macro_f1"] >= baseline["macro_f1"]]
    return min(eligible, key=lambda row: (-row["accuracy"], row["views"])) if eligible else baseline


def compare(features, rows) -> dict:
    labels = np.array([row["label"] for row in rows])
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(rows, labels))
    results = []
    for views in VIEW_COUNTS:
        start = time.perf_counter()
        scores = []
        predicted = np.empty(len(labels), dtype=object)
        for fold, (train_idx, val_idx) in enumerate(folds):
            x, y = training_rows(features, labels, train_idx, views)
            classifier = build_classifier()
            classifier.fit(x, y)
            pred = classifier.predict(features[val_idx, 0])
            predicted[val_idx] = pred
            score = metrics(labels[val_idx], pred)
            score.update(fold=fold, train_originals=len(train_idx), train_samples=len(y), validation_originals=len(val_idx))
            scores.append(score)
            print(f"views={views} fold={fold}: accuracy={score['accuracy']:.4f} F1={score['macro_f1']:.4f}", flush=True)
        results.append({
            "views": views, "originals": len(rows), "augmented_samples": len(rows) * views,
            "accuracy": float(np.mean([r["accuracy"] for r in scores])),
            "macro_f1": float(np.mean([r["macro_f1"] for r in scores])),
            "accuracy_std": float(np.std([r["accuracy"] for r in scores])),
            "macro_f1_std": float(np.std([r["macro_f1"] for r in scores])),
            "folds": scores, "out_of_fold": metrics(labels, predicted),
            "predictions": [{"hash": row["hash"], "true": row["label"], "predicted": str(pred)}
                            for row, pred in zip(rows, predicted, strict=True)],
            "elapsed_seconds": time.perf_counter() - start,
        })
    return {
        "selection_data": "train_pool_only", "seed": SEED, "cv": "5-fold stratified",
        "fold_membership": [{"train_hashes": [rows[i]["hash"] for i in train],
                             "validation_hashes": [rows[i]["hash"] for i in val]} for train, val in folds],
        "normalization": "RGB; aspect-fit; /255; backbone mean/std; L2 per backbone; StandardScaler fit within training fold",
        "classifier": {"type": "logistic", "C": 0.1, "class_weight": "balanced", "max_iter": 1500},
        "results": results, "selected_views": select_result(results)["views"],
    }


def train_candidate(features, splits, comparison, batch_size, output):
    views = comparison["selected_views"]
    if views == 0:
        print("No augmentation improved accuracy without reducing macro-F1; keeping current model.", flush=True)
        return
    candidate = output / "candidate"
    candidate.mkdir(parents=True, exist_ok=True)
    rows = splits["train_pool"]
    labels = np.array([row["label"] for row in rows])
    classifier = build_classifier()
    classifier.fit(*training_rows(features, labels, np.arange(len(rows)), views))
    joblib.dump(classifier, candidate / "classifier.joblib")
    save_json(candidate / "training.json", {
        "backbone_key": "fusion", "backbones": BACKBONES, "views": views,
        "originals": len(rows), "train_samples": len(rows) * (views + 1),
        "normalization": comparison["normalization"], "classifier": comparison["classifier"],
        "seed": SEED, "selection": select_result(comparison["results"]), "audit": splits["audit"],
    })
    # Test data is first accessed only after the configuration has been selected.
    digests = sorted({row["hash"] for row in splits["test_raw"]})
    matrix = fused_views(digests, batch_size, 0)[:, 0]
    vectors = dict(zip(digests, matrix, strict=True))
    for key in ("test_raw", "test_clean"):
        score = report(key, splits[key], classifier, vectors, candidate)
        print(f"{key}: accuracy={score['accuracy']:.4f} macro-F1={score['macro_f1']:.4f}", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbones = load_models(device)
    with Image.open(IMAGES / f"{digests[0]}.jpg") as source:
        prediction = predict_fusion_image(source, joblib.load(candidate / "classifier.joblib"), backbones, device)
    np.testing.assert_allclose(list(prediction["scores"].values()), classifier.predict_proba(matrix[:1])[0], rtol=1e-3, atol=1e-4)
    prediction["overlay"].save(candidate / "smoke_overlay.png")
    save_json(candidate / "smoke_inference.json", {k: v for k, v in prediction.items() if k != "overlay"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--train-candidate", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.output_dir.resolve() == (ARTIFACTS / "fusion").resolve():
        parser.error("Use a separate experiment output directory")
    start = time.perf_counter()
    splits = load_json(DATA / "splits.json")
    digests = [row["hash"] for row in splits["train_pool"]]
    features = fused_views(digests, args.batch_size, max(VIEW_COUNTS))
    comparison = compare(features, splits["train_pool"])
    comparison["extraction_and_cv_seconds"] = time.perf_counter() - start
    comparison["cache_keys"] = {key: cache_key(cache_metadata(digests, key, max(VIEW_COUNTS))) for key in BACKBONE_KEYS}
    save_json(args.output_dir / "comparison.json", comparison)
    if args.train_candidate:
        train_candidate(features, splits, comparison, args.batch_size, args.output_dir)


if __name__ == "__main__":
    main()
