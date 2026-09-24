"""Final held-out evaluation of the CV-selected frozen-feature fusion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "artifacts" / "fusion_candidate.json"
CLASSES = ["ampliacao", "assimetria_lateral", "espaco_da_lingua", "normal", "queixo_para_baixo", "queixo_para_cima"]


def main() -> None:
    split = json.loads((DATA / "splits.json").read_text(encoding="utf-8"))
    selected = json.loads((ROOT / "artifacts" / "feature_comparison.json").read_text(encoding="utf-8"))["results"][0]
    assert (selected["features"], selected["algorithm"], selected["c"], selected["class_weight"]) == ("fusion:all", "logistic", 0.1, "balanced")
    digests = sorted({row["hash"] for key in ("train_pool", "test_raw") for row in split[key]})
    vectors = []
    for backbone in ("rad-dino", "dinov2-small"):
        with np.load(DATA / f"features_{backbone}_v2.npz") as saved:
            vectors.append(Normalizer().fit_transform(np.stack([saved[digest] for digest in digests])))
    fused = dict(zip(digests, np.concatenate(vectors, axis=1), strict=True))

    train = split["train_pool"]
    x_train = np.stack([fused[row["hash"]] for row in train])
    y_train = [row["label"] for row in train]
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=1500))
    model.fit(x_train, y_train)

    result = {"selection": selected, "train_unique": len(train), "test_used_for_training": False}
    for name in ("test_raw", "test_clean"):
        rows = split[name]
        actual = [row["label"] for row in rows]
        x = np.stack([fused[row["hash"]] for row in rows])
        predicted = model.predict(x).tolist()
        result[name] = {
            "n": len(rows),
            "accuracy": float(accuracy_score(actual, predicted)),
            "macro_f1": float(f1_score(actual, predicted, labels=CLASSES, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(actual, predicted, labels=CLASSES).tolist(),
            "classification_report": classification_report(actual, predicted, labels=CLASSES, zero_division=0, output_dict=True),
            "predictions": [{"hash": row["hash"], "true": true, "predicted": pred} for row, true, pred in zip(rows, actual, predicted, strict=True)],
        }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    for name in ("test_raw", "test_clean"):
        print(name, "accuracy", round(result[name]["accuracy"], 4), "macro_f1", round(result[name]["macro_f1"], 4))


if __name__ == "__main__":
    main()
