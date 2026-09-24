"""Cross-validate light task adaptation using cached inputs to the final ViT block."""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch import nn

from adapted_model import AdaptedHead
from experiments.compare_augmentation import metrics
from fusion_model import report
from pipeline import (
    ARTIFACTS,
    BACKBONES,
    CLASSES,
    DATA,
    IMAGES,
    image_to_tensor,
    load_backbone,
    load_json,
    save_json,
)

OUTPUT = ARTIFACTS / "adapted"
EPOCHS = (10, 20, 30)


def cached_tokens(rows, backbone, device):
    digests = [r["hash"] for r in rows]
    fingerprint = hashlib.sha256((str(digests) + BACKBONES["dinov2-small"]["revision"] + "518x252-v1").encode()).hexdigest()
    cache = DATA / f"penultimate_{fingerprint}.npy"
    if cache.exists():
        return np.load(cache)
    chunks = []
    for start in range(0, len(rows), 4):
        images = []
        for row in rows[start:start + 4]:
            with Image.open(IMAGES / f"{row['hash']}.jpg") as image:
                images.append(image_to_tensor(image, "dinov2-small"))
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=device == "cuda"):
            output = backbone(pixel_values=torch.stack(images).to(device), output_hidden_states=True)
        chunks.append(output.hidden_states[-2].cpu().numpy().astype(np.float16))
        if start % 100 == 0:
            print(f"tokens {min(start + 4, len(rows))}/{len(rows)}", flush=True)
    array = np.concatenate(chunks)
    np.save(cache, array)
    return array


def predict(network, tokens, indices, batch_size=16):
    network.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            with torch.autocast(device_type="cuda", enabled=tokens.device.type == "cuda"):
                logits = network(tokens[indices[start:start + batch_size]].float())
            scores.append(logits.float().softmax(dim=-1).cpu().numpy())
    return np.concatenate(scores)


def fit(backbone, tokens, labels, train_idx, epochs, callback=None):
    torch.manual_seed(42)
    np.random.seed(42)
    network = AdaptedHead(backbone).to(tokens.device)
    optimizer = torch.optim.AdamW([
        {"params": network.block.parameters(), "lr": 1e-5},
        {"params": network.norm.parameters(), "lr": 1e-5},
        {"params": network.head.parameters(), "lr": 5e-4},
    ], weight_decay=.05)
    counts = torch.bincount(labels[train_idx], minlength=len(CLASSES)).float()
    loss_fn = nn.CrossEntropyLoss(weight=(counts.sum() / (len(CLASSES) * counts)).to(tokens.device), label_smoothing=.05)
    scaler = torch.amp.GradScaler("cuda", enabled=tokens.device.type == "cuda")
    rng = np.random.default_rng(42)
    for epoch in range(1, epochs + 1):
        network.train()
        shuffled = rng.permutation(train_idx)
        loss_sum = 0.
        for start in range(0, len(shuffled), 16):
            ids = shuffled[start:start + 16]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=tokens.device.type == "cuda"):
                loss = loss_fn(network(tokens[ids].float()), labels[ids])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(network.parameters(), 1.)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach())
        if callback and epoch in EPOCHS:
            callback(network, epoch)
        if epoch % 10 == 0:
            print(f"epoch={epoch} loss={loss_sum / ((len(train_idx) + 15) // 16):.4f}", flush=True)
    return network


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate-selected", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    start = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = load_json(DATA / "splits.json")
    rows = splits["train_pool"]
    y = np.array([CLASSES.index(r["label"]) for r in rows])
    backbone = load_backbone(device, "dinov2-small")
    tokens = torch.from_numpy(cached_tokens(rows, backbone, device)).to(device)
    labels = torch.tensor(y, device=device)
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(rows, y))
    oof = {epoch: np.empty((len(rows), len(CLASSES))) for epoch in EPOCHS}
    fold_scores = {epoch: [] for epoch in EPOCHS}
    for fold, (train_idx, val_idx) in enumerate(folds):
        def record(network, epoch, val_idx=val_idx, fold=fold):
            scores = predict(network, tokens, val_idx)
            oof[epoch][val_idx] = scores
            score = metrics(np.array(CLASSES)[y[val_idx]], np.array(CLASSES)[scores.argmax(axis=1)])
            fold_scores[epoch].append(score)
            print(f"fold={fold} epoch={epoch}: accuracy={score['accuracy']:.4f}, F1={score['macro_f1']:.4f}", flush=True)
            save_json(OUTPUT / "progress.json", fold_scores)
        network = fit(backbone, tokens, labels, train_idx, max(EPOCHS), record)
        del network
    results = [{"epochs": epoch, "accuracy": float(np.mean([r["accuracy"] for r in fold_scores[epoch]])),
                "macro_f1": float(np.mean([r["macro_f1"] for r in fold_scores[epoch]])),
                "folds": fold_scores[epoch], "oof_scores": oof[epoch].tolist()} for epoch in EPOCHS]
    baseline = load_json(ARTIFACTS / "larger_backbone" / "comparison.json")["baseline"]
    eligible = [r for r in results if r["accuracy"] > baseline["accuracy"] and r["macro_f1"] >= baseline["macro_f1"]]
    selected = max(eligible, key=lambda r: (r["accuracy"], r["macro_f1"])) if eligible else None
    save_json(OUTPUT / "comparison.json", {"selection_data": "train_pool_only", "cv": "5-fold seed=42",
              "train_count": len(rows), "source_hashes": [r["hash"] for r in rows], "results": results,
              "selected_epochs": selected["epochs"] if selected else None, "seconds": time.perf_counter() - start})
    print("SELECTED", {k: selected[k] for k in ("epochs", "accuracy", "macro_f1")} if selected else None, flush=True)
    if not selected or not args.evaluate_selected:
        return
    network = fit(backbone, tokens, labels, np.arange(len(rows)), selected["epochs"])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.save(network.state_dict(), OUTPUT / "adapted.pt")
    save_json(OUTPUT / "training.json", {"backbone_key": "dinov2-small", "feature_mode": "all",
              "architecture": "adapted_last_block", "backbones": [{"key": "dinov2-small", **BACKBONES["dinov2-small"]}],
              "epochs": selected["epochs"], "train_count": len(rows), "selection": selected,
              "seed": 42, "audit": splits["audit"], "lr_block": 1e-5, "lr_head": 5e-4,
              "weight_decay": .05, "dropout": .3, "label_smoothing": .05})
    test_rows = list({r["hash"]: r for r in splits["test_raw"]}.values())
    test_tokens = cached_tokens(test_rows, backbone, device)
    vectors = dict(zip([r["hash"] for r in test_rows], test_tokens, strict=True))
    class Predictor:
        def __init__(self):
            self.named_steps = {"logisticregression": SimpleNamespace(classes_=np.array(CLASSES))}

        def predict_proba(self, matrix):
            data = torch.from_numpy(matrix).to(device)
            return predict(network, data, np.arange(len(data)))
    for key in ("test_raw", "test_clean"):
        score = report(key, splits[key], Predictor(), vectors, OUTPUT)
        print(f"{key}: accuracy={score['accuracy']:.4f}, F1={score['macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
