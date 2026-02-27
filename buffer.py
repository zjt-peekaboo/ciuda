# buffer.py
"""
Contribution-driven Memory Buffer for E3P.
Stores samples that are reliable AND informative (near the 'sweet spot'
distance from prototype, not too close, not too far).
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from loguru import logger
import numpy as np


class ContributionBuffer:
    """
    Memory buffer that stores samples based on:
    C(x) = R(x) * exp(-|cos(h(x), p_k) - m|)
    where m is the sweet-spot cosine similarity target.
    Also enforces diversity via herding-like selection.
    """

    def __init__(
        self,
        total_capacity: int = 300,
        sweet_spot_m: float = 0.65,
        feature_dim: int = 256,
    ):
        self.total_capacity = total_capacity
        self.sweet_spot_m = sweet_spot_m
        self.feature_dim = feature_dim

        # Storage: class_id -> list of (image_tensor, feature, mapped_label, score)
        self.storage: Dict[int, List[Dict]] = {}
        self.seen_classes: List[int] = []

    def _per_class_budget(self) -> int:
        """Compute per-class budget based on number of seen classes."""
        if len(self.seen_classes) == 0:
            return self.total_capacity
        return max(1, self.total_capacity // len(self.seen_classes))

    @torch.no_grad()
    def compute_contribution_scores(
        self,
        features: torch.Tensor,
        reliability: torch.Tensor,
        prototype: torch.Tensor,
    ) -> torch.Tensor:
        """
        C(x) = R(x) * exp(-|cos(h(x), p_k) - m|)
        Args:
            features: (N, D)
            reliability: (N,)
            prototype: (D,)
        Returns:
            scores: (N,)
        """
        feat_norm = F.normalize(features, dim=1)
        proto_norm = F.normalize(prototype.unsqueeze(0), dim=1)
        cos_sim = torch.mm(feat_norm, proto_norm.t()).squeeze(1)  # (N,)
        contribution = reliability * torch.exp(-torch.abs(cos_sim - self.sweet_spot_m))
        return contribution

    def update_class(
        self,
        class_id: int,
        images: torch.Tensor,
        features: torch.Tensor,
        labels: torch.Tensor,
        reliability: torch.Tensor,
        prototype: torch.Tensor,
    ):
        """
        Update buffer for a specific class.
        Args:
            class_id: mapped class index
            images: (N, C, H, W)
            features: (N, D)
            labels: (N,) all should be class_id
            reliability: (N,)
            prototype: (D,) prototype for this class
        """
        if class_id not in self.seen_classes:
            self.seen_classes.append(class_id)
            self.seen_classes.sort()

        budget = self._per_class_budget()

        # Compute contribution scores
        scores = self.compute_contribution_scores(features, reliability, prototype)

        # Select top-budget samples by contribution score
        n_select = min(budget, len(scores))
        if n_select == 0:
            return

        top_indices = torch.topk(scores, n_select).indices

        # Diversity: from top candidates, do simple herding
        selected = self._herding_select(
            features[top_indices], images[top_indices],
            labels[top_indices], scores[top_indices],
            max_select=budget,
        )

        self.storage[class_id] = selected

        # Rebalance: trim other classes if total exceeds capacity
        self._rebalance()

    def _herding_select(
        self,
        features: torch.Tensor,
        images: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
        max_select: int,
    ) -> List[Dict]:
        """Simple herding: iteratively pick sample closest to running mean."""
        N = features.size(0)
        if N <= max_select:
            return [
                {"image": images[i].cpu(), "feature": features[i].cpu(),
                 "label": labels[i].item(), "score": scores[i].item()}
                for i in range(N)
            ]

        feat_norm = F.normalize(features, dim=1)
        mean_feat = feat_norm.mean(dim=0)
        selected_indices = []
        selected_sum = torch.zeros_like(mean_feat)

        for _ in range(max_select):
            # Pick sample
# buffer.py (continued)
            # that makes running mean closest to overall mean
            candidate_means = (selected_sum.unsqueeze(0) + feat_norm) / (len(selected_indices) + 1)
            dists = torch.norm(candidate_means - mean_feat.unsqueeze(0), dim=1)
            # Mask already selected
            for idx in selected_indices:
                dists[idx] = float("inf")
            best = dists.argmin().item()
            selected_indices.append(best)
            selected_sum += feat_norm[best]

        return [
            {"image": images[i].cpu(), "feature": features[i].cpu(),
             "label": labels[i].item(), "score": scores[i].item()}
            for i in selected_indices
        ]

    def _rebalance(self):
        """Trim all classes to per-class budget if total exceeds capacity."""
        budget = self._per_class_budget()
        for cid in list(self.storage.keys()):
            if len(self.storage[cid]) > budget:
                # Keep top-scoring
                self.storage[cid].sort(key=lambda x: x["score"], reverse=True)
                self.storage[cid] = self.storage[cid][:budget]

    def get_all_samples(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return all buffered samples as (images, labels) tensors.
        Returns:
            images: (M, C, H, W)
            labels: (M,)
        """
        all_images = []
        all_labels = []
        for cid, samples in self.storage.items():
            for s in samples:
                all_images.append(s["image"])
                all_labels.append(s["label"])

        if len(all_images) == 0:
            return torch.zeros(0), torch.zeros(0, dtype=torch.long)

        images = torch.stack(all_images, dim=0)
        labels = torch.tensor(all_labels, dtype=torch.long)
        return images, labels

    def get_class_samples(self, class_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return buffered samples for a specific class."""
        if class_id not in self.storage or len(self.storage[class_id]) == 0:
            return torch.zeros(0), torch.zeros(0, dtype=torch.long)
        imgs = torch.stack([s["image"] for s in self.storage[class_id]], dim=0)
        lbls = torch.tensor([s["label"] for s in self.storage[class_id]], dtype=torch.long)
        return imgs, lbls

    def total_stored(self) -> int:
        return sum(len(v) for v in self.storage.values())

    def log_status(self):
        per_class = {c: len(v) for c, v in self.storage.items()}
        logger.info(
            f"Buffer: {self.total_stored()}/{self.total_capacity} samples, "
            f"{len(self.seen_classes)} classes, per-class: {per_class}"
        )
