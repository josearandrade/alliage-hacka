"""Compare frozen-feature classifiers without consulting the held-out test set."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "artifacts" / "feature_comparison.json"
LABELS = [
    "ampliacao", "assimetria_lateral", "espaco_da_lingua", "normal",
    "queixo_para_baixo", "queixo_para_cima",
]
GROUPS = {"mean": (1,), "spatial": (2, 3, 4, 5, 6, 7), "mean_spatial": (1, 2, 3, 4, 5, 6, 7), "all": tuple(range(8))}


def load_matrix(backbone: str, digests: list[str], mode: str) -> np.ndarray:
    with np.load(DATA / f"features_{backbone}_v2.npz") as saved:
        matrix = np.stack([saved[digest] for digest in digests]).astype(np.float32)
    width = matrix.shape[1] // 8
    return np.concatenate([matrix[:, i * width : (i + 1) * width] for i in GROUPS[mode]], axis=1)


def build_model(algorithm: str, c: float, weight: str | None, pca_components: int | None):
    estimator = (
        LogisticRegression(C=c, class_weight=weight, max_iter=1500)
        if algorithm == "logistic"
        else SVC(C=c, gamma="scale", class_weight=weight, random_state=42)
    )
    steps = [StandardScaler()]
    if pca_components:
        steps.append(PCA(n_components=pca_components, random_state=42))
    steps.append(estimator)
    return make_pipeline(*steps)


def main() -> None:
    splits = json.loads((DATA / "splits.json").read_text(encoding="utf-8"))
    rows = splits["train_pool"]
    digests = [row["hash"] for row in rows]
    y = np.array([row["label"] for row in rows])
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(digests, y))
    matrices = {}
    for backbone in ("rad-dino", "dinov2-small"):
        for mode in ("mean", "spatial", "mean_spatial", "all"):
            matrices[f"{backbone}:{mode}"] = load_matrix(backbone, digests, mode)
    # Give each backbone unit-length contribution before concatenation.
    for mode in ("mean", "spatial", "mean_spatial", "all"):
        a = Normalizer().fit_transform(matrices[f"rad-dino:{mode}"])
        b = Normalizer().fit_transform(matrices[f"dinov2-small:{mode}"])
        matrices[f"fusion:{mode}"] = np.concatenate([a, b], axis=1)

    results = []
    for feature_key, x in matrices.items():
        for algorithm, cs in (("logistic", (0.01, 0.1, 1.0)), ("svm", (0.1, 1.0, 10.0))):
            for weight in (None, "balanced"):
                for c in cs:
                    scores = []
                    accuracies = []
                    for train_idx, val_idx in folds:
                        model = build_model(algorithm, c, weight, None)
                        model.fit(x[train_idx], y[train_idx])
                        pred = model.predict(x[val_idx])
                        scores.append(f1_score(y[val_idx], pred, labels=LABELS, average="macro", zero_division=0))
                        accuracies.append(accuracy_score(y[val_idx], pred))
                    result = {
                        "features": feature_key,
                        "algorithm": algorithm,
                        "c": c,
                        "class_weight": weight,
                        "macro_f1": float(np.mean(scores)),
                        "macro_f1_std": float(np.std(scores)),
                        "accuracy": float(np.mean(accuracies)),
                        "fold_macro_f1": [float(score) for score in scores],
                    }
                    results.append(result)
                    print(f"{feature_key:28s} {algorithm:8s} C={c:<4g} weight={weight!s:8s} F1={result['macro_f1']:.4f}", flush=True)

    results.sort(key=lambda row: (-row["macro_f1"], row["macro_f1_std"]))
    OUT.write_text(json.dumps({"selection_data": "train_pool_only", "cv": "5-fold stratified, seed=42", "n": len(rows), "results": results}, indent=2), encoding="utf-8")
    print("TOP 10")
    for row in results[:10]:
        print({key: row[key] for key in ("features", "algorithm", "c", "class_weight", "macro_f1", "macro_f1_std")})


if __name__ == "__main__":
    main()
