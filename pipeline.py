"""Prepare, train, evaluate and run inference for the panoramic positioning demo."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from transformers import AutoModel

ROOT = Path(__file__).resolve().parent
ZIP_PATH = ROOT / "desafio 2-20260924T171536Z-1-001.zip"
DATA = ROOT / "data"
IMAGES = DATA / "images"
ARTIFACTS = ROOT / "artifacts"
EXPERIMENTS = ARTIFACTS / "experiments.json"
BACKBONES = {
    "dinov2-small": {
        "id": "facebook/dinov2-small",
        "revision": "ed25f3a31f01632728cabb09d1542f84ab7b0056",
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
    "rad-dino": {
        "id": "microsoft/rad-dino",
        "revision": "110cbc18d5133582e320b43d53bf5c44e410c936",
        "mean": (0.5307, 0.5307, 0.5307),
        "std": (0.2583, 0.2583, 0.2583),
    },
}
IMAGE_SIZE = (518, 252)  # width, height; both divisible by the DINOv2 patch size (14)
SEED = 42
CV_SPLITS = 5
PATCH_ROWS = IMAGE_SIZE[1] // 14
PATCH_COLS = IMAGE_SIZE[0] // 14
FEATURE_GROUPS = {
    "mean": (1,),
    "mean_cls": (1, 0),
    "spatial": (2, 3, 4, 5, 6, 7),
    "mean_spatial": (1, 2, 3, 4, 5, 6, 7),
    "all": (0, 1, 2, 3, 4, 5, 6, 7),
}

CLASSES = [
    "ampliacao",
    "assimetria_lateral",
    "espaco_da_lingua",
    "normal",
    "queixo_para_baixo",
    "queixo_para_cima",
]
CLASS_NAMES = {
    "ampliacao": "Ampliação",
    "assimetria_lateral": "Assimetria lateral",
    "espaco_da_lingua": "Espaço da língua",
    "normal": "Normal",
    "queixo_para_baixo": "Queixo para baixo",
    "queixo_para_cima": "Queixo para cima",
}
ALIASES = {
    "ampliacao": "ampliacao",
    "amplificacao": "ampliacao",
    "assimetria_lateral": "assimetria_lateral",
    "assimetria": "assimetria_lateral",
    "espaco_da_lingua": "espaco_da_lingua",
    "espaco_lingua": "espaco_da_lingua",
    "normal": "normal",
    "queixo_para_baixo": "queixo_para_baixo",
    "queixo_para_cima": "queixo_para_cima",
}


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def make_splits(records: list[dict]) -> dict:
    """Remove contradictory training labels and every test image from training."""
    by_hash = defaultdict(list)
    for row in records:
        by_hash[row["hash"]].append(row)

    test_hashes = {row["hash"] for row in records if row["set"] == "teste"}
    train_rows = []
    conflicting_train = set()
    for digest, rows in by_hash.items():
        labels = {row["label"] for row in rows if row["set"] == "dataset"}
        if len(labels) > 1:
            conflicting_train.add(digest)
        if len(labels) == 1 and digest not in test_hashes:
            train_rows.append({"hash": digest, "label": next(iter(labels))})

    raw_test = [row for row in records if row["set"] == "teste"]
    clean_test = [
        rows[0]
        for digest, group in by_hash.items()
        if (rows := [row for row in group if row["set"] == "teste"])
        and len({row["label"] for row in rows}) == 1
    ]
    assert not ({row["hash"] for row in train_rows} & test_hashes)
    assert {row["label"] for row in train_rows} == set(CLASSES)
    assert {row["label"] for row in clean_test} == set(CLASSES)
    return {
        "train_pool": sorted(train_rows, key=lambda row: row["hash"]),
        "test_raw": raw_test,
        "test_clean": sorted(clean_test, key=lambda row: row["hash"]),
        "audit": {
            "dataset_files": sum(row["set"] == "dataset" for row in records),
            "test_files": len(raw_test),
            "dataset_conflicting_hashes": len(conflicting_train),
            "cross_split_hashes": len(test_hashes & {row["hash"] for row in records if row["set"] == "dataset"}),
            "train_pool_unique": len(train_rows),
            "test_clean_unique": len(clean_test),
            "test_conflicting_hashes": sum(
                len({row["label"] for row in group if row["set"] == "teste"}) > 1
                for group in by_hash.values()
            ),
            "train_pool_by_class": dict(Counter(row["label"] for row in train_rows)),
            "test_raw_by_class": dict(Counter(row["label"] for row in raw_test)),
            "test_clean_by_class": dict(Counter(row["label"] for row in clean_test)),
        },
    }


def prepare() -> dict:
    IMAGES.mkdir(parents=True, exist_ok=True)
    records = []
    with zipfile.ZipFile(ZIP_PATH) as outer:
        for outer_entry, set_name in (
            ("desafio 2/dataset.zip", "dataset"),
            ("desafio 2/teste.zip", "teste"),
        ):
            with zipfile.ZipFile(io.BytesIO(outer.read(outer_entry))) as inner:
                for member in inner.infolist():
                    if member.is_dir() or not member.filename.lower().endswith(".jpg"):
                        continue
                    parts = member.filename.replace("\\", "/").split("/")
                    if len(parts) != 3 or parts[0] != set_name or parts[1] not in ALIASES:
                        raise ValueError("Estrutura inesperada no ZIP de imagens")
                    image_bytes = inner.read(member)
                    digest = hashlib.sha256(image_bytes).hexdigest()
                    image_path = IMAGES / f"{digest}.jpg"
                    if not image_path.exists():
                        image_path.write_bytes(image_bytes)
                    records.append({"hash": digest, "label": ALIASES[parts[1]], "set": set_name})
    splits = make_splits(records)
    save_json(DATA / "splits.json", splits)
    print(json.dumps(splits["audit"], ensure_ascii=False, indent=2))
    return splits


def image_to_tensor(image: Image.Image, backbone_key: str) -> torch.Tensor:
    # Fit the full panoramic image into a 2:1 canvas; never crop anatomy.
    image = image.convert("RGB")
    image.thumbnail(IMAGE_SIZE, Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", IMAGE_SIZE, (0, 0, 0))
    canvas.paste(image, ((IMAGE_SIZE[0] - image.width) // 2, (IMAGE_SIZE[1] - image.height) // 2))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    config = BACKBONES[backbone_key]
    array = (array - np.array(config["mean"], dtype=np.float32)) / np.array(config["std"], dtype=np.float32)
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


def augment_image(image: Image.Image, seed: int) -> Image.Image:
    """Apply label-preserving acquisition changes for panoramic radiographs."""
    rng = np.random.default_rng(seed)
    result = image.convert("RGB")
    result = ImageEnhance.Brightness(result).enhance(float(rng.uniform(0.88, 1.12)))
    result = ImageEnhance.Contrast(result).enhance(float(rng.uniform(0.85, 1.15)))
    if rng.random() < 0.5:
        result = ImageEnhance.Sharpness(result).enhance(float(rng.uniform(0.7, 1.3)))
    if rng.random() < 0.25:
        result = result.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.2, 0.8))))
    array = np.asarray(result, dtype=np.float32)
    noise = rng.normal(0.0, 2.0, size=array.shape).astype(np.float32)
    return Image.fromarray(np.uint8(np.clip(array + noise, 0, 255)))


def load_backbone(device: str, backbone_key: str):
    config = BACKBONES[backbone_key]
    model = AutoModel.from_pretrained(
        config["id"],
        revision=config["revision"],
        use_safetensors=True,
        cache_dir=DATA / "hf_cache",
    )
    model.eval().to(device)
    return model


@torch.inference_mode()
def extract_batch(model, images: list[Image.Image], device: str, backbone_key: str) -> tuple[np.ndarray, np.ndarray]:
    batch = torch.stack([image_to_tensor(image, backbone_key) for image in images]).to(device)
    with torch.autocast(device_type="cuda", enabled=device == "cuda"):
        tokens = model(pixel_values=batch).last_hidden_state
    tokens = tokens.float().cpu().numpy()
    patches = tokens[:, 1:, :]
    grid = patches.reshape(len(images), PATCH_ROWS, PATCH_COLS, -1)
    boundaries = np.linspace(0, PATCH_COLS, 4, dtype=int)
    regions = [
        grid[:, y0:y1, boundaries[col]:boundaries[col + 1]].mean(axis=(1, 2))
        for y0, y1 in ((0, PATCH_ROWS // 2), (PATCH_ROWS // 2, PATCH_ROWS))
        for col in range(3)
    ]
    combined = np.concatenate([tokens[:, 0, :], patches.mean(axis=1), *regions], axis=1)
    return combined, patches


def select_features(matrix: np.ndarray, mode: str) -> np.ndarray:
    width = matrix.shape[-1] // 8
    groups = FEATURE_GROUPS[mode]
    return np.concatenate([matrix[..., group * width : (group + 1) * width] for group in groups], axis=-1)


def model_scores(classifier, matrix: np.ndarray) -> np.ndarray:
    """Return comparable class scores; SVM scores are relative, not calibrated."""
    if "logisticregression" in classifier.named_steps:
        return classifier.predict_proba(matrix)
    decisions = classifier.decision_function(matrix)
    shifted = decisions - decisions.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def extract_all(splits: dict, batch_size: int, backbone_key: str) -> dict[str, np.ndarray]:
    digests = sorted({row["hash"] for key in ("train_pool", "test_raw") for row in splits[key]})
    cache = DATA / f"features_{backbone_key}_v2.npz"
    if cache.exists():
        with np.load(cache) as saved:
            if set(saved.files) == set(digests):
                return {key: saved[key] for key in saved.files}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Extraindo {len(digests)} imagens com {backbone_key} em {device}.", flush=True)
    model = load_backbone(device, backbone_key)
    features = {}
    for start in range(0, len(digests), batch_size):
        batch_ids = digests[start : start + batch_size]
        images = []
        for digest in batch_ids:
            with Image.open(IMAGES / f"{digest}.jpg") as image:
                images.append(image.copy())
        vectors, _ = extract_batch(model, images, device, backbone_key)
        features.update(zip(batch_ids, vectors, strict=True))
        if start % (batch_size * 10) == 0:
            print(f"  {min(start + batch_size, len(digests))}/{len(digests)}", flush=True)
    np.savez_compressed(cache, **features)
    return features


def extract_augmented_features(
    digests: list[str], batch_size: int, backbone_key: str
) -> dict[str, np.ndarray]:
    """Extract one deterministic augmented view per image for training only."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Extraindo augmentations de {len(digests)} imagens com {backbone_key} em {device}.", flush=True)
    model = load_backbone(device, backbone_key)
    augmented = {}
    for start in range(0, len(digests), batch_size):
        batch_ids = digests[start : start + batch_size]
        images = []
        for digest in batch_ids:
            with Image.open(IMAGES / f"{digest}.jpg") as image:
                images.append(augment_image(image, SEED + int(digest[:8], 16)))
        vectors, _ = extract_batch(model, images, device, backbone_key)
        augmented.update(zip(batch_ids, vectors, strict=True))
    return augmented


def report_set(name: str, rows: list[dict], classifier, features: dict, feature_mode: str) -> dict:
    truth = [row["label"] for row in rows]
    matrix = select_features(np.stack([features[row["hash"]] for row in rows]), feature_mode)
    probabilities = model_scores(classifier, matrix)
    label_order = classifier.steps[-1][1].classes_.tolist()
    predicted = [label_order[index] for index in probabilities.argmax(axis=1)]
    cm = confusion_matrix(truth, predicted, labels=CLASSES)
    confusions = sorted(
        (
            {"real": CLASSES[i], "predicted": CLASSES[j], "count": int(cm[i, j])}
            for i in range(len(CLASSES))
            for j in range(len(CLASSES))
            if i != j and cm[i, j] > 0
        ),
        key=lambda row: (-row["count"], row["real"], row["predicted"]),
    )
    result = {
        "n": len(rows),
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=CLASSES, average="macro", zero_division=0)),
        "classification_report": classification_report(
            truth, predicted, labels=CLASSES, output_dict=True, zero_division=0
        ),
        "confusion_matrix": cm.tolist(),
        "confusions": confusions,
        "predictions": [
            {
                "hash": row["hash"],
                "true": actual,
                "predicted": guess,
                "confidence": float(max(scores)),
                "scores": {label: float(scores[label_order.index(label)]) for label in CLASSES},
            }
            for row, actual, guess, scores in zip(rows, truth, predicted, probabilities, strict=True)
        ],
    }
    save_json(ARTIFACTS / f"{name}.json", result)
    fig, ax = plt.subplots(figsize=(8, 7))
    counts = np.array(result["confusion_matrix"])
    ax.imshow(counts, cmap="Blues")
    for (i, j), value in np.ndenumerate(counts):
        ax.text(j, i, str(value), ha="center", va="center", color="white" if value > counts.max() / 2 else "black")
    short = ["Ampliação", "Assimetria", "Língua", "Normal", "Queixo ↓", "Queixo ↑"]
    ax.set_xticks(range(6), short, rotation=35, ha="right")
    ax.set_yticks(range(6), short)
    ax.set_xlabel("Predita")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusão — {name} (n={len(rows)})")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / f"{name}_confusion.png", dpi=150)
    plt.close(fig)
    print(f"{name}: n={len(rows)}, acurácia={result['accuracy']:.3f}, macro-F1={result['macro_f1']:.3f}")
    return result


def make_classifier(algorithm: str, c: float, class_weight: str | None):
    estimator = (
        LogisticRegression(C=c, class_weight=class_weight, max_iter=1000)
        if algorithm == "logistic"
        else SVC(C=c, gamma="scale", class_weight=class_weight, random_state=SEED)
    )
    return make_pipeline(StandardScaler(), estimator)


def cross_validate_choices(
    pool: list[dict],
    features: dict[str, np.ndarray],
    augmented: dict[str, np.ndarray] | None,
) -> list[dict]:
    labels = np.array([row["label"] for row in pool])
    splitter = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)
    choices = []
    augmentation_modes = ("none", "augmented") if augmented is not None else ("none",)
    for augmentation in augmentation_modes:
        for mode in FEATURE_GROUPS:
            for class_weight in (None, "balanced"):
                for algorithm, c_values in (("logistic", (0.01, 0.1, 1.0)), ("svm", (0.1, 1.0, 10.0))):
                    for c in c_values:
                        fold_scores = []
                        for train_index, val_index in splitter.split(pool, labels):
                            train_rows = [pool[index] for index in train_index]
                            val_rows = [pool[index] for index in val_index]
                            train_vectors = [features[row["hash"]] for row in train_rows]
                            train_labels = [row["label"] for row in train_rows]
                            if augmentation == "augmented":
                                train_vectors += [augmented[row["hash"]] for row in train_rows]
                                train_labels += train_labels.copy()
                            train_x = select_features(np.stack(train_vectors), mode)
                            val_x = select_features(np.stack([features[row["hash"]] for row in val_rows]), mode)
                            model = make_classifier(algorithm, c, class_weight)
                            model.fit(train_x, train_labels)
                            predictions = model.predict(val_x)
                            fold_scores.append(float(f1_score(
                                [row["label"] for row in val_rows],
                                predictions,
                                labels=CLASSES,
                                average="macro",
                                zero_division=0,
                            )))
                        choices.append({
                            "score": float(np.mean(fold_scores)),
                            "std": float(np.std(fold_scores)),
                            "fold_scores": fold_scores,
                            "augmentation": augmentation,
                            "mode": mode,
                            "algorithm": algorithm,
                            "class_weight": class_weight,
                            "c": c,
                        })
    return choices


def train(batch_size: int, backbone_key: str, validate_only: bool = False) -> None:
    splits = load_json(DATA / "splits.json") if (DATA / "splits.json").exists() else prepare()
    features = extract_all(splits, batch_size, backbone_key)
    pool = splits["train_pool"]
    augmented = extract_augmented_features([row["hash"] for row in pool], batch_size, backbone_key)
    choices = cross_validate_choices(pool, features, augmented)
    for choice in choices:
        print(
            f"CV: {choice['augmentation']}, {choice['mode']}, {choice['algorithm']}, "
            f"peso={choice['class_weight']}, C={choice['c']:g}, "
            f"macro-F1={choice['score']:.3f} +/- {choice['std']:.3f}",
            flush=True,
        )
    best = max(choices, key=lambda item: (item["score"], -item["std"], -item["c"], -len(FEATURE_GROUPS[item["mode"]])))
    best_c = best["c"]
    feature_mode = best["mode"]
    print(f"Selecionado: {backbone_key}, {feature_mode}, {best['algorithm']}, augment={best['augmentation']}, peso={best['class_weight']}, C={best_c:g}")
    if validate_only:
        return
    final = make_classifier(best["algorithm"], best_c, best["class_weight"])
    all_vectors = [features[row["hash"]] for row in pool]
    all_labels = [row["label"] for row in pool]
    if best["augmentation"] == "augmented":
        all_vectors += [augmented[row["hash"]] for row in pool]
        all_labels += all_labels.copy()
    final.fit(select_features(np.stack(all_vectors), feature_mode), all_labels)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(final, ARTIFACTS / "classifier.joblib")
    save_json(ARTIFACTS / "training.json", {
        "model_id": BACKBONES[backbone_key]["id"],
        "model_revision": BACKBONES[backbone_key]["revision"],
        "backbone_key": backbone_key,
        "image_size": IMAGE_SIZE,
        "seed": SEED,
        "train_count": len(pool),
        "validation_count": 0,
        "final_fit_count": len(pool),
        "selected_c": best_c,
        "feature_mode": feature_mode,
        "classifier_type": best["algorithm"],
        "class_weight": best["class_weight"],
        "augmentation": best["augmentation"],
        "tta": best["augmentation"] == "augmented",
        "validation": {"type": "stratified_kfold", "splits": CV_SPLITS, "seed": SEED},
        "validation_scores": choices,
        "audit": splits["audit"],
    })
    report_set("test_raw", splits["test_raw"], final, features, feature_mode)
    report_set("test_clean", splits["test_clean"], final, features, feature_mode)
    benchmark_inference(final, splits["test_clean"][0]["hash"], feature_mode, backbone_key)


def benchmark_experiments(batch_size: int) -> None:
    """Compare frozen backbones under the same cross-validation protocol."""
    splits = load_json(DATA / "splits.json") if (DATA / "splits.json").exists() else prepare()
    pool = splits["train_pool"]
    results = []
    for backbone_key in BACKBONES:
        features = extract_all(splits, batch_size, backbone_key)
        choices = cross_validate_choices(pool, features, None)
        best = max(choices, key=lambda item: (item["score"], -item["std"], -item["c"]))
        results.append({
            "backbone_key": backbone_key,
            "model_id": BACKBONES[backbone_key]["id"],
            "best": best,
            "top_choices": sorted(choices, key=lambda item: item["score"], reverse=True)[:10],
        })
        print(f"Benchmark {backbone_key}: macro-F1={best['score']:.3f} +/- {best['std']:.3f}", flush=True)
    save_json(EXPERIMENTS, {
        "seed": SEED,
        "validation": {"type": "stratified_kfold", "splits": CV_SPLITS},
        "train_count": len(pool),
        "results": results,
    })


def benchmark_inference(classifier, digest: str, feature_mode: str, backbone_key: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = load_backbone(device, backbone_key)
    with Image.open(IMAGES / f"{digest}.jpg") as source:
        image = source.copy()
    for _ in range(3):
        predict_image(image, classifier, backbone, device, feature_mode, backbone_key)
    samples = [predict_image(image, classifier, backbone, device, feature_mode, backbone_key)["elapsed_ms"] for _ in range(20)]
    result = {
        "device": device,
        "warmup_runs": 3,
        "measured_runs": 20,
        "median_ms": float(np.median(samples)),
        "p95_ms": float(np.percentile(samples, 95)),
        "includes": "preprocessing, backbone, classifier and heatmap",
    }
    save_json(ARTIFACTS / "inference_benchmark.json", result)
    print(f"Inferência: mediana={result['median_ms']:.1f} ms, p95={result['p95_ms']:.1f} ms ({device})")


def predict_image(
    image: Image.Image,
    classifier,
    backbone,
    device: str,
    feature_mode: str,
    backbone_key: str,
    tta: bool = False,
):
    start = time.perf_counter()
    full_vector, patches = extract_batch(backbone, [image], device, backbone_key)
    vector = select_features(full_vector, feature_mode)
    scores = model_scores(classifier, vector)[0]
    if tta:
        augmented_images = [augment_image(image, SEED + offset) for offset in (101, 202)]
        augmented_vectors, _ = extract_batch(backbone, augmented_images, device, backbone_key)
        augmented_scores = model_scores(
            classifier, select_features(augmented_vectors, feature_mode)
        )
        scores = np.mean(np.vstack([scores, augmented_scores]), axis=0)
    order = classifier.steps[-1][1].classes_
    ranked = np.argsort(scores)[::-1]
    winner, runner_up = int(ranked[0]), int(ranked[1])
    if "logisticregression" in classifier.named_steps:
        heat = linear_patch_heatmap(classifier, patches[0], winner, runner_up, feature_mode)
    else:
        heat = svm_occlusion_heatmap(classifier, full_vector[0], patches[0], winner, runner_up, feature_mode)
    heat = np.maximum(heat, 0)
    if heat.max() > heat.min():
        heat = (heat - heat.min()) / (heat.max() - heat.min())
    heat_image = Image.fromarray(np.uint8(heat * 255)).resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    config = BACKBONES[backbone_key]
    base = Image.fromarray(np.uint8(np.clip((image_to_tensor(image, backbone_key).numpy().transpose(1, 2, 0) *
        np.array(config["std"]) + np.array(config["mean"])) * 255, 0, 255)))
    colors = plt.get_cmap("inferno")(np.asarray(heat_image, dtype=np.float32) / 255.0)[:, :, :3]
    overlay = Image.blend(base, Image.fromarray(np.uint8(colors * 255)), 0.38)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000
    return {
        "label": str(order[winner]),
        "confidence": float(scores[winner]),
        "scores": {str(label): float(score) for label, score in zip(order, scores, strict=True)},
        "overlay": overlay,
        "elapsed_ms": elapsed_ms,
    }


def linear_patch_heatmap(classifier, patches: np.ndarray, winner: int, runner_up: int, feature_mode: str):
    scaler = classifier.named_steps["standardscaler"]
    linear = classifier.named_steps["logisticregression"]
    margin_weights = linear.coef_[winner] - linear.coef_[runner_up]
    width = patches.shape[-1]
    heat = np.zeros((PATCH_ROWS, PATCH_COLS), dtype=np.float32)
    boundaries = np.linspace(0, PATCH_COLS, 4, dtype=int)
    for position, group in enumerate(FEATURE_GROUPS[feature_mode]):
        if group == 0:
            continue
        weights = margin_weights[position * width : (position + 1) * width]
        average = scaler.mean_[position * width : (position + 1) * width]
        scale = scaler.scale_[position * width : (position + 1) * width]
        if group == 1:
            heat += (((patches - average) / scale) @ weights).reshape(PATCH_ROWS, PATCH_COLS) / len(patches)
        else:
            region = group - 2
            y0, y1 = ((0, PATCH_ROWS // 2), (PATCH_ROWS // 2, PATCH_ROWS))[region // 3]
            x0, x1 = boundaries[region % 3], boundaries[region % 3 + 1]
            patch_region = patches.reshape(PATCH_ROWS, PATCH_COLS, width)[y0:y1, x0:x1]
            heat[y0:y1, x0:x1] += ((patch_region - average) / scale @ weights) / ((y1-y0) * (x1-x0))
    return heat


def svm_occlusion_heatmap(classifier, full_vector: np.ndarray, patches: np.ndarray,
                          winner: int, runner_up: int, feature_mode: str) -> np.ndarray:
    """Approximate regional importance by neutralizing fixed patch embeddings."""
    grid = patches.reshape(PATCH_ROWS, PATCH_COLS, -1)
    width = patches.shape[-1]
    global_mean = patches.mean(axis=0)
    candidates = []
    cells = []
    for y0 in range(0, PATCH_ROWS, 3):
        y1 = min(y0 + 3, PATCH_ROWS)
        for x0 in range(0, PATCH_COLS, 4):
            x1 = min(x0 + 4, PATCH_COLS)
            candidate = full_vector.copy()
            area = grid[y0:y1, x0:x1]
            candidate[width:2 * width] += (global_mean - area.mean(axis=(0, 1))) * area.shape[0] * area.shape[1] / len(patches)
            bounds = np.linspace(0, PATCH_COLS, 4, dtype=int)
            for region in range(6):
                ry0, ry1 = ((0, PATCH_ROWS // 2), (PATCH_ROWS // 2, PATCH_ROWS))[region // 3]
                rx0, rx1 = bounds[region % 3], bounds[region % 3 + 1]
                oy0, oy1 = max(y0, ry0), min(y1, ry1)
                ox0, ox1 = max(x0, rx0), min(x1, rx1)
                if oy0 < oy1 and ox0 < ox1:
                    affected = grid[oy0:oy1, ox0:ox1]
                    count = affected.shape[0] * affected.shape[1]
                    total = (ry1 - ry0) * (rx1 - rx0)
                    start = (region + 2) * width
                    candidate[start:start + width] += (global_mean - affected.mean(axis=(0, 1))) * count / total
            candidates.append(candidate)
            cells.append((y0, y1, x0, x1))
    candidate_scores = model_scores(classifier, select_features(np.stack(candidates), feature_mode))
    base_scores = model_scores(classifier, select_features(full_vector[None, :], feature_mode))[0]
    base_margin = base_scores[winner] - base_scores[runner_up]
    heat = np.zeros((PATCH_ROWS, PATCH_COLS), dtype=np.float32)
    for scores, (y0, y1, x0, x1) in zip(candidate_scores, cells, strict=True):
        heat[y0:y1, x0:x1] = base_margin - (scores[winner] - scores[runner_up])
    return heat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "benchmark", "train"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--backbone", choices=BACKBONES, default="dinov2-small")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "benchmark":
        benchmark_experiments(args.batch_size)
    else:
        train(args.batch_size, args.backbone, args.validate_only)


if __name__ == "__main__":
    main()
