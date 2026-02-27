# energy_reliability.py
"""
Energy-gap + Multi-view Agreement based reliability scoring.
Core of E3P: replaces QSI's invariance quantification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from loguru import logger


class ReliabilityEstimator:
    """
    Compute per-sample reliability R(x) = sigmoid(delta_E(x)) * A(x)
    where:
      - delta_E = top1_logit - top2_logit (energy gap)
      - A = pairwise agreement ratio across multiple augmented views
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    @torch.no_grad()
    def compute_energy_gap(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute energy gap: difference between top-1 and top-2 logits.
        Args:
            logits: (B, C)
        Returns:
            delta_e: (B,) energy gap per sample
        """
        top2_vals, _ = torch.topk(logits, k=min(2, logits.size(1)), dim=1)
        if logits.size(1) < 2:
            return top2_vals[:, 0]
        delta_e = top2_vals[:, 0] - top2_vals[:, 1]
        return delta_e

    @torch.no_grad()
    def compute_agreement(
        self,
        predictions_list: list,
    ) -> torch.Tensor:
        """
        Compute pairwise agreement ratio across multiple views.
        Args:
            predictions_list: list of (B,) tensors, each is argmax prediction
        Returns:
            agreement: (B,) in [0, 1]
        """
        n_views = len(predictions_list)
        if n_views < 2:
            return torch.ones(predictions_list[0].shape[0], device=predictions_list[0].device)

        B = predictions_list[0].shape[0]
        n_pairs = 0
        agree_count = torch.zeros(B, device=predictions_list[0].device)

        for i in range(n_views):
            for j in range(i + 1, n_views):
                agree_count += (predictions_list[i] == predictions_list[j]).float()
                n_pairs += 1

        agreement = agree_count / max(n_pairs, 1)
        return agreement

    @torch.no_grad()
    def compute_reliability(
        self,
        model: nn.Module,
        views: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute reliability R(x) for a batch.
        Args:
            model: E3PModel
            views: dict with keys 'weak', 'strong', 'test'
            device: torch device
        Returns:
            reliability: (B,) reliability scores
            pseudo_labels: (B,) pseudo labels from test view
            features: (B, D) features from test view
        """
        model.eval()

        # View 1: weak augmentation
        logits_weak, _ = model(views["weak"].to(device))
        pred_weak = logits_weak.argmax(dim=1)

        # View 2: strong augmentation
        logits_strong, _ = model(views["strong"].to(device))
        pred_strong = logits_strong.argmax(dim=1)

        # View 3 & 4: test view with dropout perturbation
        model.train()  # enable dropout
        logits_drop1, _ = model(views["test"].to(device))
        pred_drop1 = logits_drop1.argmax(dim=1)
        logits_drop2, _ = model(views["test"].to(device))
        pred_drop2 = logits_drop2.argmax(dim=1)
        model.eval()

        # Main prediction from test view (clean)
        logits_test, features = model(views["test"].to(device))
        pseudo_labels = logits_test.argmax(dim=1)

        # Energy gap from test view logits
        delta_e = self.compute_energy_gap(logits_test)

        # Agreement across 4 views
        agreement = self.compute_agreement([pred_weak, pred_strong, pred_drop1, pred_drop2])

        # R(x) = sigmoid(delta_E) * A(x)
        reliability = torch.sigmoid(delta_e / self.temperature) * agreement

        return reliability, pseudo_labels, features

    @torch.no_grad()
    def compute_reliability_simple(
        self,
        model: nn.Module,
        images: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simplified reliability for single-view input (e.g., buffer samples).
        Uses energy gap only (no multi-view).
        """
        model.eval()
        logits, features = model(images.to(device))
        pseudo_labels = logits.argmax(dim=1)
        delta_e = self.compute_energy_gap(logits)
        reliability = torch.sigmoid(delta_e / self.temperature)
        return reliability, pseudo_labels, features
