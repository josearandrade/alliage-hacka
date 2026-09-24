"""Compare independent spatial-block normalization without changing labels."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from experiments.compare_augmentation import metrics
from experiments.compare_larger_backbone import load_features
from fusion_model import normalize_feature_blocks, report, unit
from pipeline import ARTIFACTS, BACKBONES, DATA, load_json, save_json

OUTPUT = ARTIFACTS / "region_normalization"
KEYS = ("rad-dino", "dinov2-small")
WIDTHS = (768, 384)


def build(c, balanced):
    return make_pipeline(FunctionTransformer(normalize_feature_blocks, kw_args={"widths": WIDTHS}),
                         StandardScaler(), LogisticRegression(C=c, class_weight="balanced" if balanced else None, max_iter=2000))


def evaluate(matrix, y, folds, c, balanced):
    results = []
    predictions = np.empty(len(y), dtype=object)
    for train, val in folds:
        model = build(c, balanced)
        model.fit(matrix[train], y[train])
        predictions[val] = model.predict(matrix[val])
        results.append(metrics(y[val], predictions[val]))
    return {"accuracy": float(np.mean([r["accuracy"] for r in results])),
            "macro_f1": float(np.mean([r["macro_f1"] for r in results])), "folds": results,
            "predictions": predictions.tolist()}


def main():
    splits = load_json(DATA / "splits.json")
    rows = splits["train_pool"]
    y = np.array([r["label"] for r in rows])
    matrix = np.concatenate([unit(load_features(rows, key, 2)) for key in KEYS], axis=-1)
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(rows, y))
    results = []
    for balanced in (True, False):
        for c in (.01, .1, 1., 10.):
            with threadpool_limits(limits=4):
                result = evaluate(matrix, y, folds, c, balanced)
            result.update(c=c, balanced=balanced)
            results.append(result)
            print(f"group norm C={c} balanced={balanced}: accuracy={result['accuracy']:.4f} F1={result['macro_f1']:.4f}", flush=True)
    baseline = load_json(ARTIFACTS / "larger_backbone" / "comparison.json")["baseline"]
    eligible = [r for r in results if r["accuracy"] > baseline["accuracy"] and r["macro_f1"] >= baseline["macro_f1"]]
    selected = max(eligible, key=lambda r: (r["accuracy"], r["macro_f1"])) if eligible else None
    summary = {"selection_data": "train_pool_only", "cv": "5-fold seed=42", "results": results,
               "selected": selected, "baseline": baseline, "source_hashes": [r["hash"] for r in rows]}
    save_json(OUTPUT / "comparison.json", summary)
    print("SELECTED", {k: selected[k] for k in ("c", "balanced", "accuracy", "macro_f1")} if selected else None, flush=True)
    if not selected:
        return
    folds17 = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=17).split(rows, y))
    with threadpool_limits(limits=4):
        summary["confirmation_seed17"] = evaluate(matrix, y, folds17, selected["c"], selected["balanced"])
    save_json(OUTPUT / "comparison.json", summary)
    model = build(selected["c"], selected["balanced"])
    with threadpool_limits(limits=4):
        model.fit(matrix, y)
    candidate = OUTPUT / "candidate"
    candidate.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, candidate / "classifier.joblib")
    save_json(candidate / "training.json", {
        "backbone_key": "fusion", "feature_mode": "all",
        "backbones": [{"key": key, **BACKBONES[key]} for key in KEYS],
        "normalization": "L2 per spatial/global block; StandardScaler",
        "classifier_type": "logistic", "selected_c": selected["c"],
        "class_weight": "balanced" if selected["balanced"] else None,
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
