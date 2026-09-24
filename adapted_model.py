"""DINOv2-small with only its final transformer block adapted to the task."""

from __future__ import annotations

import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn

from pipeline import (
    CLASSES,
    IMAGE_SIZE,
    PATCH_COLS,
    PATCH_ROWS,
    image_to_tensor,
    load_backbone,
)


def pool_tokens(tokens):
    patches = tokens[:, 1:]
    grid = patches.reshape(len(tokens), PATCH_ROWS, PATCH_COLS, -1)
    bounds = np.linspace(0, PATCH_COLS, 4, dtype=int)
    regions = [grid[:, y0:y1, bounds[col]:bounds[col + 1]].mean(dim=(1, 2))
               for y0, y1 in ((0, PATCH_ROWS // 2), (PATCH_ROWS // 2, PATCH_ROWS)) for col in range(3)]
    return torch.cat([tokens[:, 0], patches.mean(dim=1), *regions], dim=-1)


class AdaptedHead(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.block = copy.deepcopy(backbone.encoder.layer[-1])
        self.norm = copy.deepcopy(backbone.layernorm)
        self.feature_norm = nn.LayerNorm(backbone.config.hidden_size * 8, elementwise_affine=False)
        self.dropout = nn.Dropout(.3)
        self.head = nn.Linear(backbone.config.hidden_size * 8, len(CLASSES))
        nn.init.normal_(self.head.weight, std=.01)
        nn.init.zeros_(self.head.bias)

    def from_final_tokens(self, tokens):
        return self.head(self.dropout(self.feature_norm(pool_tokens(tokens))))

    def forward(self, tokens):
        return self.from_final_tokens(self.norm(self.block(tokens)))


def load_adapted(directory, device):
    backbone = load_backbone(device, "dinov2-small")
    network = AdaptedHead(backbone).to(device)
    network.load_state_dict(torch.load(directory / "adapted.pt", map_location=device, weights_only=True))
    network.eval()
    backbone.encoder.layer[-1].load_state_dict(network.block.state_dict())
    backbone.layernorm.load_state_dict(network.norm.state_dict())
    backbone.eval()
    return backbone, network


def predict_adapted_image(image, models, device):
    start = time.perf_counter()
    backbone, network = models
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=device == "cuda"):
        tokens = backbone(pixel_values=image_to_tensor(image, "dinov2-small")[None].to(device)).last_hidden_state
    # Gradient-times-activation is an approximate model explanation, not clinical localization.
    with torch.enable_grad():
        tokens = tokens.float().detach().requires_grad_(True)
        logits = network.from_final_tokens(tokens)
        scores = logits.softmax(dim=-1)[0]
        rank = scores.argsort(descending=True)
        gradient = torch.autograd.grad(logits[0, rank[0]] - logits[0, rank[1]], tokens)[0]
        heat = (gradient[:, 1:] * tokens[:, 1:]).sum(dim=-1).detach().cpu().numpy().reshape(PATCH_ROWS, PATCH_COLS)
    scores = scores.detach().cpu().numpy()
    heat = np.maximum(heat, 0)
    heat = heat / max(float(heat.max()), 1e-12)
    heat_image = Image.fromarray(np.uint8(heat * 255)).resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    base = image.convert("RGB")
    base.thumbnail(IMAGE_SIZE, Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", IMAGE_SIZE)
    canvas.paste(base, ((IMAGE_SIZE[0] - base.width) // 2, (IMAGE_SIZE[1] - base.height) // 2))
    colors = plt.get_cmap("inferno")(np.asarray(heat_image) / 255.)[:, :, :3]
    overlay = Image.blend(canvas, Image.fromarray(np.uint8(colors * 255)), .38)
    return {"label": CLASSES[int(scores.argmax())], "confidence": float(scores.max()),
            "scores": dict(zip(CLASSES, map(float, scores), strict=True)), "overlay": overlay,
            "elapsed_ms": (time.perf_counter() - start) * 1000}
