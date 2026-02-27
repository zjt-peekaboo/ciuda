# prototype.py
"""
Prototype Library with EMA update and prototype-guided pseudo-label correction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from loguru import logger


class PrototypeLibrary:
    """
    Maintains per-class prototypes in feature space.
    Updated via EMA using only high-reliability samples.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        ema_beta: float = 0.9,
        temperature: float = 0.07,
        reliability_threshold: float = 0.3,
        device: str = "cuda",
    ):
        self.feature_dim = feature_dim
        self.ema_beta = ema_beta
        self.temperature = temperature
        self.reliability_threshold = reliability_threshold
        self.device = torch.device(device)

        # Prototype storage: class_id -> normalized feature vector
        self.prototypes: Dict[int, torch.Tensor] = {}
        # Count of updates per class (for logging)
        self.update_counts: Dict[int, int] = {}

    def has_class(self, class_id: int) -> bool:
        return class_id in self.prototypes

    def get_active_classes(self) -> List[int]:
        return sorted(self.prototypes.keys())

    def get_prototype_matrix(self) -> Tuple[torch.Tensor, List[int]]:
        """
        Returns:
            proto_matrix: (K, D) matrix of prototypes
            class_ids: list of K class ids
        """
        class_ids = self.get_active_classes()
        if len(class_ids) == 0:
            return torch.zeros(0, self.feature_dim, device=self.device), []
        proto_matrix = torch.stack([self.prototypes[c] for c in class_ids], dim=0)
        return proto_matrix, class_ids

    @torch.no_grad()
    def update(
        self,
        features: torch.Tensor,
        pseudo_labels: torch.Tensor,
        reliability: torch.Tensor,
    ):
        """
        Update prototypes using high-reliability samples via EMA.
        Args:
            features: (N, D)
            pseudo_labels: (N,)
            reliability: (N,)
        """
        mask = reliability > self.reliability_threshold
        if mask.sum() == 0:
            return

        feat_sel = features[mask]
        labels_sel = pseudo_labels[mask]

        for c in labels_sel.unique().tolist():
            c_mask = labels_sel == c
            c_feats = feat_sel[c_mask]
            c_mean = F.normalize(c_feats.mean(dim=0), dim=0)

            if c not in self.prototypes:
                self.prototypes[c] = c_mean.to(self.device)
                self.update_counts[c] = c_feats.size(0)
            else:
                old = self.prototypes[c]
                self.prototypes[c] = F.normalize(
                    self.ema_beta * old + (1 - self.ema_beta) * c_mean, dim=0
                )
                self.update_counts[c] += c_feats.size(0)

    @torch.no_grad()
    def correct_pseudo_labels(
        self,
        features: torch.Tensor,
        model_probs: torch.Tensor,
        reliability: torch.Tensor,
        num_classes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prototype-guided pseudo-label correction.
        hat_p(x) = R(x)*p(x) + (1-R(x))*q(x)
        where q(x) = softmax(cosine_sim(h(x), prototypes) / tau)

        Args:
            features: (N, D) normalized features
            model_probs: (N, C) softmax probabilities from model
            reliability: (N,) reliability scores
            num_classes: total number of source classes
        Returns:
            corrected_labels: (N,) corrected pseudo-labels
            corrected_probs: (N, C) corrected probability distribution
        """
        proto_matrix, class_ids = self.get_prototype_matrix()

        if len(class_ids) == 0:
            # No prototypes yet, fall back to model predictions
            return model_probs.argmax(dim=1), model_probs

        # Cosine similarity to prototypes
        feat_norm = F.normalize(features, dim=1)  # (N, D)
        sim = torch.mm(feat_norm, proto_matrix.t()) / self.temperature  # (N, K)

        # Build full probability over all classes (fill zeros for unseen)
        q = torch.zeros_like(model_probs)  # (N, C)
        sim_softmax = F.softmax(sim, dim=1)  # (N, K)
        for i, c in enumerate(class_ids):
            if c < q.size(1):
                q[:, c] = sim_softmax[:, i]

        # Normalize q (in case not all classes have prototypes)
        q_sum = q.sum(dim=1, keepdim=True).clamp(min=1e-8)
        q = q / q_sum

        # Fuse: hat_p = R * p + (1-R) * q
        R = reliability.unsqueeze(1)  # (N, 1)
        corrected_probs = R * model_probs + (1 - R) * q

        corrected_labels = corrected_probs.argmax(dim=1)
        return corrected_labels, corrected_probs

    def log_status(self):
        logger.info(
            f"Prototype library: {len(self.prototypes)} classes, "
            f"updates: {dict(sorted(self.update_counts.items()))}"
        )
