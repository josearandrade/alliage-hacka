"""Selected two-backbone classifier and approximate visual explanation."""

from __future__ import annotations

import argparse
import time
from types import SimpleNamespace

import joblib
import matplotlib.pyplot as plt
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline import (
    ARTIFACTS,
    BACKBONES,
    CLASSES,
    DATA,
    IMAGE_SIZE,
    IMAGES,
    PATCH_COLS,
    PATCH_ROWS,
    extract_all,
    extract_batch,
    image_to_tensor,
    load_backbone,
    load_json,
    save_json,
)

FUSION_DIR = ARTIFACTS / "fusion"
BACKBONE_KEYS = ("rad-dino", "dinov2-small")


class FusionEnsemble:
    """Average independently regularized linear models on the same fused features."""

    def __init__(self, members):
        self.members = members
        self.classes_ = members[0].named_steps["logisticregression"].classes_
        for member in members:
            if not np.array_equal(member.named_steps["logisticregression"].classes_, self.classes_):
                raise ValueError("Ensemble class ordering differs")
        self.named_steps = {"logisticregression": SimpleNamespace(classes_=self.classes_)}

    def predict_proba(self, matrix):
        return np.mean([member.predict_proba(matrix) for member in self.members], axis=0)

    def margin_coefficients(self, feature, winner, runner_up):
        gradients = []
        for member in self.members:
            probabilities = member.predict_proba(feature[None])[0]
            linear = member.named_steps["logisticregression"]
            expected = probabilities @ linear.coef_
            gradient = (probabilities[winner] * (linear.coef_[winner] - expected)
                        - probabilities[runner_up] * (linear.coef_[runner_up] - expected))
            gradients.append(gradient / member.named_steps["standardscaler"].scale_)
        return np.mean(gradients, axis=0)


def unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    return np.divide(vector, norm, out=np.zeros_like(vector), where=norm > 0)


def normalize_feature_blocks(matrix: np.ndarray, widths: tuple[int, ...]) -> np.ndarray:
    """Normalize each spatial/global block independently, preserving its coordinates."""
    output = []
    start = 0
    for width in widths:
        branch = matrix[:, start:start + 8 * width].reshape(len(matrix), 8, width)
        output.append(unit(branch).reshape(len(matrix), -1))
        start += 8 * width
    if start != matrix.shape[-1]:
        raise ValueError("Feature widths do not match the configured backbones")
    return np.concatenate(output, axis=-1)


def fused_features(vectors: dict[str, dict[str, np.ndarray]], digest: str) -> np.ndarray:
    return np.concatenate([unit(vectors[key][digest]) for key in BACKBONE_KEYS])


def load_models(device: str, backbone_keys=BACKBONE_KEYS) -> dict:
    models = {}
    for entry in backbone_keys:
        key = entry["key"] if isinstance(entry, dict) else entry
        model = load_backbone(device, key)
        if isinstance(entry, dict) and "image_size" in entry:
            model._alliage_image_size = tuple(entry["image_size"])
        models[key] = model
    return models


def explain(classifier, full_vectors: list[np.ndarray], patch_vectors: list[np.ndarray], winner: int, runner_up: int) -> np.ndarray:
    """Approximate spatial part of the winning-vs-runner-up linear margin."""
    if isinstance(classifier, FusionEnsemble):
        feature = np.concatenate([unit(full) for full in full_vectors])
        margin = classifier.margin_coefficients(feature, winner, runner_up)
        scaler = SimpleNamespace(scale_=None)
    else:
        scaler = classifier.named_steps["standardscaler"]
        linear = classifier.named_steps["logisticregression"]
        margin = linear.coef_[winner] - linear.coef_[runner_up]
    heat = np.zeros((PATCH_ROWS, PATCH_COLS), dtype=np.float32)
    branch_start = 0
    for full, patches in zip(full_vectors, patch_vectors, strict=True):
        factor = round(np.sqrt(len(patches) / (PATCH_ROWS * PATCH_COLS)))
        rows, cols = PATCH_ROWS * factor, PATCH_COLS * factor
        x_bounds = np.linspace(0, cols, 4, dtype=int)
        branch_heat = np.zeros((rows, cols), dtype=np.float32)
        width = patches.shape[-1]
        norm = max(float(np.linalg.norm(full)), np.finfo(np.float32).tiny)
        grid = patches.reshape(rows, cols, width)
        for group in range(1, 8):
            start = branch_start + group * width
            if "functiontransformer" in classifier.named_steps:
                norm = max(float(np.linalg.norm(full[group * width:(group + 1) * width])), np.finfo(np.float32).tiny)
            scale = scaler.scale_[start : start + width] if scaler.scale_ is not None else 1.0
            weights = margin[start : start + width] / scale / norm
            if group == 1:
                branch_heat += (grid @ weights) / (rows * cols)
            else:
                region = group - 2
                y0, y1 = ((0, rows // 2), (rows // 2, rows))[region // 3]
                x0, x1 = x_bounds[region % 3], x_bounds[region % 3 + 1]
                branch_heat[y0:y1, x0:x1] += (grid[y0:y1, x0:x1] @ weights) / ((y1 - y0) * (x1 - x0))
        heat += branch_heat.reshape(PATCH_ROWS, factor, PATCH_COLS, factor).sum(axis=(1, 3))
        branch_start += 8 * width
    return heat


def predict_fusion_image(image: Image.Image, classifier, backbones: dict, device: str) -> dict:
    start = time.perf_counter()
    full_vectors = []
    patch_vectors = []
    for key, backbone in backbones.items():
        full, patches = extract_batch(backbone, [image], device, key)
        full_vectors.append(full[0])
        patch_vectors.append(patches[0])
    feature = np.concatenate([unit(vector) for vector in full_vectors])[None, :]
    scores = classifier.predict_proba(feature)[0]
    labels = classifier.named_steps["logisticregression"].classes_
    rank = np.argsort(scores)[::-1]
    winner, runner_up = int(rank[0]), int(rank[1])
    heat = np.maximum(explain(classifier, full_vectors, patch_vectors, winner, runner_up), 0)
    if heat.max() > heat.min():
        heat = (heat - heat.min()) / (heat.max() - heat.min())
    heat_image = Image.fromarray(np.uint8(heat * 255)).resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    config = BACKBONES["rad-dino"]
    base = Image.fromarray(np.uint8(np.clip(
        (image_to_tensor(image, "rad-dino").numpy().transpose(1, 2, 0) *
         np.array(config["std"]) + np.array(config["mean"])) * 255, 0, 255,
    )))
    colors = plt.get_cmap("inferno")(np.asarray(heat_image, dtype=np.float32) / 255.0)[:, :, :3]
    overlay = Image.blend(base, Image.fromarray(np.uint8(colors * 255)), 0.38)
    if device == "cuda":
        torch.cuda.synchronize()
    return {
        "label": str(labels[winner]),
        "confidence": float(scores[winner]),
        "scores": {str(label): float(score) for label, score in zip(labels, scores, strict=True)},
        "overlay": overlay,
        "elapsed_ms": (time.perf_counter() - start) * 1000,
    }


def report(name: str, rows: list[dict], classifier, vectors: dict[str, np.ndarray], output_dir=FUSION_DIR) -> dict:
    actual = [row["label"] for row in rows]
    matrix = np.stack([vectors[row["hash"]] for row in rows])
    scores = classifier.predict_proba(matrix)
    labels = classifier.named_steps["logisticregression"].classes_.tolist()
    predicted = [labels[index] for index in scores.argmax(axis=1)]
    cm = confusion_matrix(actual, predicted, labels=CLASSES)
    result = {
        "n": len(rows),
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=CLASSES, average="macro", zero_division=0)),
        "classification_report": classification_report(actual, predicted, labels=CLASSES, output_dict=True, zero_division=0),
        "confusion_matrix": cm.tolist(),
        "confusions": sorted(
            ({"real": CLASSES[i], "predicted": CLASSES[j], "count": int(cm[i, j])}
             for i in range(6) for j in range(6) if i != j and cm[i, j] > 0),
            key=lambda row: (-row["count"], row["real"], row["predicted"]),
        ),
        "predictions": [
            {"hash": row["hash"], "true": true, "predicted": pred,
             "confidence": float(max(prob)),
             "scores": {label: float(prob[labels.index(label)]) for label in CLASSES}}
            for row, true, pred, prob in zip(rows, actual, predicted, scores, strict=True)
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / f"{name}.json", result)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cm, cmap="Blues")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center", color="white" if value > cm.max() / 2 else "black")
    short = ["Ampliação", "Assimetria", "Língua", "Normal", "Queixo ↓", "Queixo ↑"]
    ax.set_xticks(range(6), short, rotation=35, ha="right")
    ax.set_yticks(range(6), short)
    ax.set_xlabel("Predita")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusão — {name} (n={len(rows)})")
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}_confusion.png", dpi=150)
    plt.close(fig)
    return result


def train(batch_size: int) -> None:
    splits = load_json(DATA / "splits.json")
    selection = load_json(ARTIFACTS / "feature_comparison.json")["results"][0]
    assert (selection["features"], selection["algorithm"], selection["c"], selection["class_weight"]) == ("fusion:all", "logistic", 0.1, "balanced")
    vectors_by_backbone = {key: extract_all(splits, batch_size, key) for key in BACKBONE_KEYS}
    digests = {row["hash"] for key in ("train_pool", "test_raw") for row in splits[key]}
    vectors = {digest: fused_features(vectors_by_backbone, digest) for digest in digests}
    pool = splits["train_pool"]
    classifier = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=1500))
    classifier.fit(np.stack([vectors[row["hash"]] for row in pool]), [row["label"] for row in pool])
    FUSION_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, FUSION_DIR / "classifier.joblib")
    save_json(FUSION_DIR / "training.json", {
        "backbone_key": "fusion", "backbones": [
            {"key": key, "id": BACKBONES[key]["id"], "revision": BACKBONES[key]["revision"]}
            for key in BACKBONE_KEYS
        ],
        "feature_mode": "all", "normalization": "L2 per backbone, then StandardScaler",
        "classifier_type": "logistic", "selected_c": 0.1, "class_weight": "balanced",
        "train_count": len(pool), "seed": 42, "validation": {"type": "stratified_kfold", "splits": 5, "seed": 42},
        "validation_score": selection["macro_f1"], "validation_std": selection["macro_f1_std"],
        "selection_candidates": len(load_json(ARTIFACTS / "feature_comparison.json")["results"]),
        "audit": splits["audit"],
    })
    for key in ("test_raw", "test_clean"):
        result = report(key, splits[key], classifier, vectors)
        print(f"{key}: n={result['n']}, accuracy={result['accuracy']:.4f}, macro-F1={result['macro_f1']:.4f}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbones = load_models(device)
    with Image.open(IMAGES / f"{splits['test_clean'][0]['hash']}.jpg") as source:
        sample = source.copy()
    for _ in range(3):
        predict_fusion_image(sample, classifier, backbones, device)
    times = [predict_fusion_image(sample, classifier, backbones, device)["elapsed_ms"] for _ in range(20)]
    save_json(FUSION_DIR / "inference_benchmark.json", {
        "device": device, "warmup_runs": 3, "measured_runs": 20,
        "median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)),
        "includes": "preprocessing, both backbones, classifier and approximate heatmap",
    })
    print(f"inference median={np.median(times):.1f} ms, p95={np.percentile(times, 95):.1f} ms ({device})", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train", choices=["train"])
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    train(args.batch_size)
