"""Compare three predetermined equal-weight ensembles on training folds only."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments.compare_augmentation import metrics
from experiments.compare_larger_backbone import load_features
from fusion_model import FusionEnsemble, report, unit
from pipeline import ARTIFACTS, BACKBONES, CLASSES, DATA, load_json, save_json

OUTPUT = ARTIFACTS / "ensemble"
KEYS = ("rad-dino", "dinov2-small")
SETTINGS = ((.01, True), (.1, True), (1., True), (.01, False), (.1, False), (1., False))
ENSEMBLES = {"regularization": (0, 1, 2), "class_balance": (1, 4), "six_models": tuple(range(6))}


def fit(matrix, labels, c, balanced):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced" if balanced else None, max_iter=2000))
    model.fit(matrix, labels)
    return model


def cross_validate(matrix, y, seed):
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(matrix, y))
    probabilities = np.zeros((len(SETTINGS), len(y), len(CLASSES)))
    for fold, (train, val) in enumerate(folds):
        for member, (c, balanced) in enumerate(SETTINGS):
            model = fit(matrix[train], y[train], c, balanced)
            probabilities[member, val] = model.predict_proba(matrix[val])
        print(f"ensemble seed={seed} fold={fold} done", flush=True)
    results = []
    for name, members in ENSEMBLES.items():
        scores = probabilities[list(members)].mean(axis=0)
        predictions = np.array(CLASSES)[scores.argmax(axis=1)]
        per_fold = [metrics(y[val], predictions[val]) for _, val in folds]
        results.append({"name": name, "members": members,
                        "accuracy": float(np.mean([r["accuracy"] for r in per_fold])),
                        "macro_f1": float(np.mean([r["macro_f1"] for r in per_fold])),
                        "folds": per_fold, "oof_scores": scores.tolist()})
    return results


def main():
    splits = load_json(DATA / "splits.json")
    rows = splits["train_pool"]
    y = np.array([r["label"] for r in rows])
    matrix = np.concatenate([unit(load_features(rows, key, 2)) for key in KEYS], axis=-1)
    with threadpool_limits(limits=4):
        results = cross_validate(matrix, y, 42)
    baseline = load_json(ARTIFACTS / "larger_backbone" / "comparison.json")["baseline"]
    for r in results:
        print(r["name"], r["accuracy"], r["macro_f1"], flush=True)
    eligible = [r for r in results if r["accuracy"] > baseline["accuracy"] and r["macro_f1"] >= baseline["macro_f1"]]
    selected = max(eligible, key=lambda r: (r["accuracy"], r["macro_f1"])) if eligible else None
    summary = {"selection_data": "train_pool_only", "cv": "5-fold seed=42", "results": results,
               "selected": selected, "baseline": baseline, "settings": SETTINGS,
               "source_hashes": [r["hash"] for r in rows]}
    save_json(OUTPUT / "comparison.json", summary)
    if not selected:
        print("No ensemble selected", flush=True)
        return
    with threadpool_limits(limits=4):
        confirmation = cross_validate(matrix, y, 17)
        summary["confirmation_seed17"] = next(r for r in confirmation if r["name"] == selected["name"])
        model = FusionEnsemble([fit(matrix, y, *SETTINGS[i]) for i in selected["members"]])
    save_json(OUTPUT / "comparison.json", summary)
    candidate = OUTPUT / "candidate"
    candidate.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, candidate / "classifier.joblib")
    save_json(candidate / "training.json", {
        "backbone_key": "fusion", "feature_mode": "all",
        "backbones": [{"key": key, **BACKBONES[key]} for key in KEYS],
        "normalization": "L2 per backbone; training-fold StandardScaler for each member",
        "classifier_type": "ensemble", "settings": [SETTINGS[i] for i in selected["members"]],
        "train_count": len(rows), "seed": 42, "validation_score": selected["macro_f1"],
        "validation_accuracy": selected["accuracy"], "selection_candidates": len(results), "audit": splits["audit"],
    })
    test_rows = list({r["hash"]: r for r in splits["test_raw"]}.values())
    test_matrix = np.concatenate([unit(load_features(test_rows, key, 2)) for key in KEYS], axis=-1)
    vectors = dict(zip([r["hash"] for r in test_rows], test_matrix, strict=True))
    for key in ("test_raw", "test_clean"):
        score = report(key, splits[key], model, vectors, candidate)
        print(f"{key}: accuracy={score['accuracy']:.4f}, F1={score['macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
