# model.py
"""ResNet50 backbone with lightweight Adapter for parameter-efficient adaptation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from loguru import logger


class Adapter(nn.Module):
    """Lightweight bottleneck adapter inserted after backbone layers."""

    def __init__(self, in_dim: int, bottleneck_dim: int = 64):
        super().__init__()
        self.down = nn.Linear(in_dim, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, in_dim)
        # Initialize near-identity
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))


class E3PModel(nn.Module):
    """
    ResNet50 feature extractor + Adapter + Classifier.
    For source training: full model trains normally.
    For target adaptation: backbone frozen, only adapter + classifier trainable.
    """

    def __init__(
        self,
        num_classes: int = 31,
        feature_dim: int = 2048,
        adapter_dim: int = 64,
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        # Backbone
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # Remove original fc
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # -> (B, 2048, 1, 1)

        # Bottleneck (optional projection)
        self.bottleneck = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        # Adapter (for target adaptation)
        self.adapter = Adapter(256, adapter_dim)

        # Classifier
        self.classifier = nn.Linear(256, num_classes)

        # Weight initialization
        nn.init.xavier_normal_(self.bottleneck[0].weight)
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def extract_features(self, x: torch.Tensor, use_adapter: bool = True) -> torch.Tensor:
        """Extract features, optionally through adapter."""
        feat = self.backbone(x).squeeze(-1).squeeze(-1)  # (B, 2048)
        feat = self.bottleneck(feat)  # (B, 256)
        if use_adapter:
            feat = self.adapter(feat)  # (B, 256)
        return feat

    def forward(self, x: torch.Tensor, use_adapter: bool = True):
        feat = self.extract_features(x, use_adapter=use_adapter)
        logits = self.classifier(feat)
        return logits, feat

    def freeze_backbone(self):
        """Freeze backbone + bottleneck, keep adapter + classifier trainable."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.bottleneck.parameters():
            param.requires_grad = False
        # Adapter and classifier remain trainable
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"Backbone frozen. Trainable: {trainable:,} / {total:,} "
            f"({100*trainable/total:.1f}%)"
        )

    def unfreeze_all(self):
        """Unfreeze all parameters (for source training)."""
        for param in self.parameters():
            param.requires_grad = True

    def get_trainable_params(self):
        """Return trainable parameters for optimizer."""
        return [p for p in self.parameters() if p.requires_grad]
