"""ResNet18 multi-head item classifier (category / gender / color)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


class ResNetMultiHead(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        in_features: int,
        num_categories: int,
        num_genders: int,
        num_colors: int,
    ):
        super().__init__()
        self.backbone = backbone
        self.fc_shared = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
        )
        self.category_head = nn.Linear(256, num_categories)
        self.gender_head = nn.Linear(256, num_genders)
        self.color_head = nn.Linear(256, num_colors)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.backbone(x)
        x = self.fc_shared(x)
        return {
            "category": self.category_head(x),
            "gender": self.gender_head(x),
            "color": self.color_head(x),
        }


def _build_model(num_categories: int, num_genders: int, num_colors: int) -> ResNetMultiHead:
    backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return ResNetMultiHead(backbone, in_features, num_categories, num_genders, num_colors)


@dataclass
class ClassifierArtifacts:
    model: ResNetMultiHead
    labels: dict[str, list[str]]
    transform: transforms.Compose
    device: torch.device


def _val_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def resolve_classifier_dir(ml_dir: Path) -> Path | None:
    env = os.getenv("CLASSIFIER_DIR")
    if env:
        p = Path(env)
        if (p / "resnet_item_classifier.pt").is_file() and (p / "label_classes.json").is_file():
            return p
    for candidate in (ml_dir / "artifacts", ml_dir / "notebooks" / "exports"):
        if (candidate / "resnet_item_classifier.pt").is_file() and (candidate / "label_classes.json").is_file():
            return candidate
    return None


def load_classifier(ml_dir: Path) -> ClassifierArtifacts | None:
    root = resolve_classifier_dir(ml_dir)
    if root is None:
        return None

    labels_path = root / "label_classes.json"
    weights_path = root / "resnet_item_classifier.pt"

    labels_raw = json.loads(labels_path.read_text(encoding="utf-8"))
    labels = {k: list(v) for k, v in labels_raw.items()}
    for key in ("category", "gender", "color"):
        if key not in labels:
            raise ValueError(f"label_classes.json missing '{key}'")

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError("Checkpoint must be a dict with 'state_dict'")
    num_cat = int(ckpt["num_categories"])
    num_gen = int(ckpt["num_genders"])
    num_col = int(ckpt["num_colors"])
    if len(labels["category"]) != num_cat or len(labels["gender"]) != num_gen or len(labels["color"]) != num_col:
        raise ValueError("label_classes.json lengths do not match checkpoint head sizes")

    model = _build_model(num_cat, num_gen, num_col)
    state = ckpt["state_dict"]
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return ClassifierArtifacts(model=model, labels=labels, transform=_val_transform(), device=device)


def predict_from_pil(art: ClassifierArtifacts, image: Image.Image) -> dict[str, Any]:
    if image.mode != "RGB":
        image = image.convert("RGB")
    x = art.transform(image).unsqueeze(0).to(art.device)
    with torch.inference_mode():
        out = art.model(x)
    result: dict[str, Any] = {}
    for head, key in (("category", "category"), ("gender", "gender"), ("color", "color")):
        logits = out[head][0]
        probs = torch.softmax(logits, dim=0)
        idx = int(torch.argmax(probs).item())
        classes = art.labels[key]
        result[key] = classes[idx]
        result[f"{key}Confidence"] = float(probs[idx].item())
    return result
