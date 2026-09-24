"""Compare whole-panorama resolution without cropping anatomy or selecting on test."""

from __future__ import annotations

import hashlib
import inspect
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from experiments.compare_larger_backbone import (
    build_classifier,
    evaluate_cv,
    load_features,
)
from fusion_model import report, unit
from pipeline import (
    ARTIFACTS,
    BACKBONES,
    DATA,
    IMAGES,
    extract_batch,
    image_to_tensor,
    load_backbone,
    load_json,
    save_json,
)

OUTPUT = ARTIFACTS / "resolution"
HIGH_SIZE = (1036, 504)
REPRESENTATIONS = {
    "high_rad": ("rad_high",),
    "high_small": ("small_high",),
    "high_fusion": ("rad_high", "small_high"),
    "high_rad_low_small": ("rad_high", "small_low"),
    "low_rad_high_small": ("rad_low", "small_high"),
}
SPECS = {
    "rad_high": {"key": "rad-dino", "image_size": HIGH_SIZE},
    "small_high": {"key": "dinov2-small", "image_size": HIGH_SIZE},
    "rad_low": {"key": "rad-dino"},
    "small_low": {"key": "dinov2-small"},
}


def high_features(rows, key):
    metadata = {"hashes": [r["hash"] for r in rows], "backbone": BACKBONES[key],
                "size": HIGH_SIZE, "implementation": inspect.getsource(image_to_tensor) + inspect.getsource(extract_batch)}
    fingerprint = hashlib.sha256(str(metadata).encode()).hexdigest()
    cache = DATA / f"highres_{key}_{fingerprint}.npz"
    if cache.exists():
        with np.load(cache) as saved:
            return saved["features"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_backbone(device, key)
    model._alliage_image_size = HIGH_SIZE
    result = []
    for index, row in enumerate(rows):
        with Image.open(IMAGES / f"{row['hash']}.jpg") as image:
            features, _ = extract_batch(model, [image], device, key)
        result.append(features[0])
        if index % 50 == 0:
            print(f"highres {key}: {index + 1}/{len(rows)}", flush=True)
    matrix = np.stack(result)
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite high-resolution features")
    np.savez_compressed(cache, features=matrix)
    save_json(cache.with_suffix(".json"), metadata)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return matrix


def features_for(rows, name):
    spec = SPECS[name]
    return unit(high_features(rows, spec["key"]) if "image_size" in spec else load_features(rows, spec["key"], 2))


def main():
    start = time.perf_counter()
    torch.set_num_threads(4)
    split = load_json(DATA / "splits.json")
    rows = split["train_pool"]
    y = np.array([r["label"] for r in rows])
    vectors = {name: features_for(rows, name) for name in SPECS}
    matrices = {name: np.concatenate([vectors[key] for key in keys], axis=-1) for name, keys in REPRESENTATIONS.items()}
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(rows, y))
    results = []
    for name, matrix in matrices.items():
        for c in (.01, .1):
            with threadpool_limits(limits=4):
                result = evaluate_cv(matrix, y, folds, c, True)
            result.update(representation=name, c=c)
            results.append(result)
            save_json(OUTPUT / "progress.json", results)
            print(f"{name} C={c}: accuracy={result['accuracy']:.4f} F1={result['macro_f1']:.4f}", flush=True)
    baseline = load_json(ARTIFACTS / "larger_backbone" / "comparison.json")["baseline"]
    eligible = [r for r in results if r["accuracy"] > baseline["accuracy"] and r["macro_f1"] >= baseline["macro_f1"]]
    selected = max(eligible, key=lambda r: (r["accuracy"], r["macro_f1"])) if eligible else None
    summary = {"selection_data": "train_pool_only", "cv": "5-fold seed=42", "results": results,
               "selected": selected, "baseline": baseline, "source_hashes": [r["hash"] for r in rows],
               "image_size": HIGH_SIZE, "elapsed_seconds": time.perf_counter() - start}
    save_json(OUTPUT / "comparison.json", summary)
    print("SELECTED", {k: selected[k] for k in ("representation", "c", "accuracy", "macro_f1")} if selected else None, flush=True)
    if not selected:
        return
    confirm = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=17).split(rows, y))
    with threadpool_limits(limits=4):
        summary["confirmation_seed17"] = evaluate_cv(matrices[selected["representation"]], y, confirm, selected["c"], True)
    save_json(OUTPUT / "comparison.json", summary)
    names = REPRESENTATIONS[selected["representation"]]
    classifier = build_classifier(selected["c"], True)
    with threadpool_limits(limits=4):
        classifier.fit(matrices[selected["representation"]], y)
    candidate = OUTPUT / "candidate"
    candidate.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, candidate / "classifier.joblib")
    save_json(candidate / "training.json", {
        "backbone_key": "fusion", "feature_mode": "all",
        "backbones": [{**BACKBONES[SPECS[name]["key"]], **SPECS[name]} for name in names],
        "normalization": "backbone pixel mean/std; L2 per backbone; training-fold StandardScaler",
        "classifier_type": "logistic", "selected_c": selected["c"], "class_weight": "balanced",
        "train_count": len(rows), "seed": 42, "validation_score": selected["macro_f1"],
        "validation_accuracy": selected["accuracy"], "selection_candidates": len(results), "audit": split["audit"],
    })
    test_rows = list({r["hash"]: r for r in split["test_raw"]}.values())
    matrix = np.concatenate([features_for(test_rows, name) for name in names], axis=-1)
    test_vectors = dict(zip([r["hash"] for r in test_rows], matrix, strict=True))
    for key in ("test_raw", "test_clean"):
        score = report(key, split[key], classifier, test_vectors, candidate)
        print(f"{key}: accuracy={score['accuracy']:.4f}, F1={score['macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
