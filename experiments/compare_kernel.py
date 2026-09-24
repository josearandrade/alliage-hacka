"""Preserve embedding geometry while comparing bounded RBF classifier settings."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from experiments.compare_augmentation import metrics
from experiments.compare_larger_backbone import load_features
from fusion_model import unit
from pipeline import ARTIFACTS, BACKBONES, DATA, load_json, model_scores, save_json

OUTPUT = ARTIFACTS / "kernel"
REPRESENTATIONS = {"rad": ("rad-dino",), "small": ("dinov2-small",), "fusion": ("rad-dino", "dinov2-small")}


def build(c, gamma):
    return make_pipeline(StandardScaler(with_std=False), SVC(
        C=c, gamma=gamma, class_weight="balanced", break_ties=True,
    ))


def evaluate(matrix, y, folds, c, gamma):
    scores = []
    predictions = np.empty(len(y), dtype=object)
    for train, val in folds:
        classifier = build(c, gamma)
        classifier.fit(matrix[train], y[train])
        predictions[val] = classifier.predict(matrix[val])
        scores.append(metrics(y[val], predictions[val]))
    return {"accuracy": float(np.mean([r["accuracy"] for r in scores])),
            "macro_f1": float(np.mean([r["macro_f1"] for r in scores])),
            "folds": scores, "predictions": predictions.tolist()}


def main():
    split = load_json(DATA / "splits.json")
    rows = split["train_pool"]
    y = np.array([r["label"] for r in rows])
    vectors = {key: unit(load_features(rows, key, 2)) for key in ("rad-dino", "dinov2-small")}
    matrices = {name: np.concatenate([vectors[key] for key in keys], axis=-1) for name, keys in REPRESENTATIONS.items()}
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(rows, y))
    results = []
    for name, matrix in matrices.items():
        for c in (1., 10.):
            for gamma in (1., 10., 100.):
                with threadpool_limits(limits=4):
                    score = evaluate(matrix, y, folds, c, gamma)
                score.update(representation=name, c=c, gamma=gamma)
                results.append(score)
                save_json(OUTPUT / "progress.json", results)
                print(f"{name} C={c} gamma={gamma}: accuracy={score['accuracy']:.4f} F1={score['macro_f1']:.4f}", flush=True)
    baseline = load_json(ARTIFACTS / "larger_backbone" / "comparison.json")["baseline"]
    eligible = [r for r in results if r["accuracy"] > baseline["accuracy"] and r["macro_f1"] >= baseline["macro_f1"]]
    selected = max(eligible, key=lambda r: (r["accuracy"], r["macro_f1"])) if eligible else None
    summary = {"selection_data": "train_pool_only", "cv": "5-fold seed=42", "results": results,
               "selected": selected, "baseline": baseline, "source_hashes": [r["hash"] for r in rows]}
    save_json(OUTPUT / "comparison.json", summary)
    print("SELECTED", {k: selected[k] for k in ("representation", "c", "gamma", "accuracy", "macro_f1")} if selected else None, flush=True)
    if not selected:
        return
    confirm = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=17).split(rows, y))
    with threadpool_limits(limits=4):
        summary["confirmation_seed17"] = evaluate(matrices[selected["representation"]], y, confirm, selected["c"], selected["gamma"])
    save_json(OUTPUT / "comparison.json", summary)
    classifier = build(selected["c"], selected["gamma"])
    classifier.fit(matrices[selected["representation"]], y)
    candidate = OUTPUT / "candidate"
    candidate.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, candidate / "classifier.joblib")
    keys = REPRESENTATIONS[selected["representation"]]
    save_json(candidate / "training.json", {
        "backbone_key": "fusion", "feature_mode": "all",
        "backbones": [{"key": key, **BACKBONES[key]} for key in keys],
        "normalization": "L2 per backbone; centered without per-coordinate scaling",
        "classifier_type": "svm", "selected_c": selected["c"], "gamma": selected["gamma"],
        "class_weight": "balanced", "train_count": len(rows), "seed": 42,
        "validation_score": selected["macro_f1"], "validation_accuracy": selected["accuracy"],
        "selection_candidates": len(results), "audit": split["audit"],
    })
    # Final evaluation only, after the selected setting is fixed.
    for key in ("test_raw", "test_clean"):
        rows = split[key]
        matrix = np.concatenate([unit(load_features(rows, backbone, 2)) for backbone in keys], axis=-1)
        probabilities = model_scores(classifier, matrix)
        labels = classifier.named_steps["svc"].classes_
        predictions = labels[probabilities.argmax(axis=1)]
        result = metrics([r["label"] for r in rows], predictions)
        result["n"] = len(rows)
        result["predictions"] = [{"hash": r["hash"], "true": r["label"], "predicted": str(p),
                                  "confidence": float(s.max()), "scores": dict(zip(labels, map(float, s), strict=True))}
                                 for r, p, s in zip(rows, predictions, probabilities, strict=True)]
        save_json(candidate / f"{key}.json", result)
        print(f"{key}: accuracy={result['accuracy']:.4f}, F1={result['macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
