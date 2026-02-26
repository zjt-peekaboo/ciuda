# utils/confidence.py
import torch
import torch.nn.functional as F
import numpy as np


class ConfidenceEstimator:
    """Estimate confidence of target samples based on Triplet distances"""
    def __init__(self, model, device, temperature=0.1):
        self.model = model
        self.device = device
        self.temperature = temperature
        
    @torch.no_grad()
    def estimate_confidence(self, features, logits, class_centers):
        """
        Estimate sample confidence based on multiple metrics
        
        Args:
            features: Feature embeddings (batch_size, feature_dim)
            logits: Model predictions (batch_size, num_classes)
            class_centers: Class center embeddings (num_classes, feature_dim)
        
        Returns:
            confidence: Sample confidence scores (batch_size,)
        """
        batch_size = features.size(0)
        
        # Get predicted labels
        probs = F.softmax(logits, dim=1)
        pred_labels = torch.argmax(probs, dim=1)
        
        # 1. Distance confidence
        dist_confidence = self._distance_confidence(features, pred_labels, class_centers)
        
        # 2. Triplet confidence
        triplet_confidence = self._triplet_confidence(features, pred_labels, class_centers)
        
        # 3. Boundary confidence
        boundary_confidence = self._boundary_confidence(features, pred_labels, class_centers)
        
        # Combine confidences
        confidence = dist_confidence * triplet_confidence * boundary_confidence
        
        return confidence
    
    def _distance_confidence(self, features, pred_labels, class_centers):
        """Confidence based on distance to predicted class center"""
        # Normalize
        features_norm = F.normalize(features, p=2, dim=1)
        centers_norm = F.normalize(class_centers, p=2, dim=1)
        
        # Get predicted class centers
        pred_centers = centers_norm[pred_labels]
        
        # Compute distance
        distances = torch.norm(features_norm - pred_centers, p=2, dim=1)
        
        # Convert to confidence
        confidence = torch.exp(-distances / self.temperature)
        
        return confidence
    
    def _triplet_confidence(self, features, pred_labels, class_centers):
        """Confidence based on triplet margin"""
        # Normalize
        features_norm = F.normalize(features, p=2, dim=1)
        centers_norm = F.normalize(class_centers, p=2, dim=1)
        
        # Distance to predicted class
        pred_centers = centers_norm[pred_labels]
        d_positive = torch.norm(features_norm - pred_centers, p=2, dim=1)
        
        # Distance to nearest other class
        all_distances = torch.cdist(features_norm.unsqueeze(0), 
                                    centers_norm.unsqueeze(0)).squeeze(0)
        
        # Mask out predicted class
        mask = torch.ones_like(all_distances, dtype=torch.bool)
        mask[torch.arange(features.size(0)), pred_labels] = False
        
        masked_distances = all_distances.masked_fill(~mask, float('inf'))
        d_negative, _ = masked_distances.min(dim=1)
        
        # Compute confidence
        margin = 0.5
        confidence = torch.sigmoid((d_negative - d_positive) / margin)
        
        return confidence
    
    def _boundary_confidence(self, features, pred_labels, class_centers):
        """Confidence based on distance to decision boundary"""
        # Normalize
        features_norm = F.normalize(features, p=2, dim=1)
        centers_norm = F.normalize(class_centers, p=2, dim=1)
        
        # Distance to all class centers
        all_distances = torch.cdist(features_norm.unsqueeze(0), 
                                    centers_norm.unsqueeze(0)).squeeze(0)
        
        # Mask out predicted class
        mask = torch.ones_like(all_distances, dtype=torch.bool)
        mask[torch.arange(features.size(0)), pred_labels] = False
        
        # Distance to nearest boundary
        masked_distances = all_distances.masked_fill(~mask, float('inf'))
        d_boundary, _ = masked_distances.min(dim=1)
        
        # Convert to confidence
        threshold = 0.5
        confidence = 1 - torch.exp(-d_boundary / threshold)
        
        return confidence


