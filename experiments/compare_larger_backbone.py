"""Bounded comparison of a larger backbone and feature scaling; train-only selection."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments.compare_augmentation import extract_views, metrics
from fusion_model import report, unit
from pipeline import ARTIFACTS, BACKBONES, DATA, load_json, save_json

OUTPUT = ARTIFACTS / "larger_backbone"
REPRESENTATIONS = {
    "control": ("rad-dino", "dinov2-small"),
    "base": ("dinov2-base",),
    "base_fusion": ("rad-dino", "dinov2-base"),
}


def load_features(rows, backbone, batch_size):
    digests = [row["hash"] for row in rows]
    cache = DATA / f"features_{backbone}_v2.npz"
    if cache.exists():
        with np.load(cache) as saved:
            if all(digest in saved for digest in digests):
                return np.stack([saved[digest] for digest in digests])
    return extract_views(digests, backbone, batch_size, 0)[:, 0]


def build_classifier(c, standardize):
    return make_pipeline(StandardScaler(with_std=standardize), LogisticRegression(
        C=c, class_weight="balanced", max_iter=2000,
    ))


def evaluate_cv(matrix, labels, folds, c, standardize):
    results = []
    predictions = np.empty(len(labels), dtype=object)
    for train_idx, val_idx in folds:
        classifier = build_classifier(c, standardize)
        classifier.fit(matrix[train_idx], labels[train_idx])
        predictions[val_idx] = classifier.predict(matrix[val_idx])
        results.append(metrics(labels[val_idx], predictions[val_idx]))
    return {
        "accuracy": float(np.mean([r["accuracy"] for r in results])),
        "macro_f1": float(np.mean([r["macro_f1"] for r in results])),
        "folds": results, "out_of_fold": metrics(labels, predictions),
        "predictions": predictions.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--evaluate-selected", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    split = load_json(DATA / "splits.json")
    rows = split["train_pool"]
    y = np.array([row["label"] for row in rows])
    keys = tuple(dict.fromkeys(key for values in REPRESENTATIONS.values() for key in values))
    vectors = {key: unit(load_features(rows, key, args.batch_size)) for key in keys}
    matrices = {name: np.concatenate([vectors[key] for key in keys], axis=-1)
                for name, keys in REPRESENTATIONS.items()}
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(rows, y))
    results = []
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, matrix in matrices.items():
        for standardize, cs in ((True, (.01, .1, 1.)), (False, (1., 10., 100.))):
            for c in cs:
                start = time.perf_counter()
                with threadpool_limits(limits=4):
                    result = evaluate_cv(matrix, y, folds, c, standardize)
                result.update(representation=name, standardize=standardize, c=c,
                              seconds=time.perf_counter() - start)
                results.append(result)
                save_json(OUTPUT / "progress.json", results)
                print(f"{name} std={standardize} C={c}: accuracy={result['accuracy']:.4f} F1={result['macro_f1']:.4f}", flush=True)
    baseline = next(r for r in results if r["representation"] == "control" and r["standardize"] and r["c"] == .1)
    eligible = [r for r in results if r["accuracy"] > baseline["accuracy"] and r["macro_f1"] >= baseline["macro_f1"]]
    selected = max(eligible, key=lambda r: (r["accuracy"], r["macro_f1"])) if eligible else baseline
    summary = {
        "selection_data": "train_pool_only", "cv": "5-fold stratified seed=42",
        "source_hashes": [row["hash"] for row in rows], "results": results,
        "selected": selected, "baseline": baseline, "candidates": len(results),
        "dinov3_access": "GatedRepoError; no weights downloaded",
        "elapsed_seconds": time.perf_counter() - started,
    }
    # A second partition measures sensitivity without selecting another configuration.
    confirm_folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=17).split(rows, y))
    with threadpool_limits(limits=4):
        summary["confirmation_seed17"] = {
            "baseline": evaluate_cv(matrices["control"], y, confirm_folds, .1, True),
            "selected": evaluate_cv(matrices[selected["representation"]], y, confirm_folds,
                                    selected["c"], selected["standardize"]),
        }
    save_json(OUTPUT / "comparison.json", summary)
    print("SELECTED", {k: selected[k] for k in ("representation", "standardize", "c", "accuracy", "macro_f1")}, flush=True)
    if not eligible or not args.evaluate_selected:
        return
    selected_keys = REPRESENTATIONS[selected["representation"]]
    classifier = build_classifier(selected["c"], selected["standardize"])
    with threadpool_limits(limits=4):
        classifier.fit(matrices[selected["representation"]], y)
    candidate = OUTPUT / "candidate"
    candidate.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, candidate / "classifier.joblib")
    save_json(candidate / "training.json", {
        "backbone_key": "fusion", "feature_mode": "all",
        "backbones": [{"key": key, **BACKBONES[key]} for key in selected_keys],
        "normalization": "L2 per backbone; centered" + (" and standardized" if selected["standardize"] else ""),
        "classifier_type": "logistic", "selected_c": selected["c"], "class_weight": "balanced",
        "train_count": len(rows), "seed": 42, "validation_score": selected["macro_f1"],
        "validation_accuracy": selected["accuracy"], "selection_candidates": len(results),
        "selection": selected, "audit": split["audit"],
    })
    # Extract held-out features only after selection and the confirmation run.
    test_rows = list({row["hash"]: row for row in split["test_raw"]}.values())
    test_matrix = np.concatenate([unit(load_features(test_rows, key, args.batch_size)) for key in selected_keys], axis=-1)
    test_vectors = dict(zip([row["hash"] for row in test_rows], test_matrix, strict=True))
    for key in ("test_raw", "test_clean"):
        score = report(key, split[key], classifier, test_vectors, candidate)
        print(f"{key}: accuracy={score['accuracy']:.4f}, macro-F1={score['macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
