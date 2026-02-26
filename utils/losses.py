# utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """Triplet loss with hard negative mining"""
    def __init__(self, margin=0.5, mining='hard'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.mining = mining
        
    def forward(self, features, labels):
        """
        Args:
            features: Feature embeddings (batch_size, feature_dim)
            labels: Class labels (batch_size,)
        """
        batch_size = features.size(0)
        
        # Compute pairwise distances
        dist_matrix = self._pairwise_distance(features)
        
        # Create mask for positive and negative pairs
        labels = labels.unsqueeze(1)
        pos_mask = labels == labels.t()
        neg_mask = labels != labels.t()
        
        # Remove self-comparisons
        pos_mask.fill_diagonal_(False)
        
        if self.mining == 'hard':
            # Hard positive mining: furthest positive
            pos_dist = dist_matrix * pos_mask.float()
            pos_dist = pos_dist.masked_fill(~pos_mask, -float('inf'))
            hardest_positive_dist, _ = pos_dist.max(dim=1)
            
            # Hard negative mining: closest negative
            neg_dist = dist_matrix.clone()
            neg_dist = neg_dist.masked_fill(~neg_mask, float('inf'))
            hardest_negative_dist, _ = neg_dist.min(dim=1)
            
        elif self.mining == 'semi-hard':
            # Semi-hard negative mining
            pos_dist = dist_matrix * pos_mask.float()
            pos_dist = pos_dist.masked_fill(~pos_mask, -float('inf'))
            hardest_positive_dist, _ = pos_dist.max(dim=1)
            
            # Select negatives that are harder than positive but within margin
            neg_dist = dist_matrix.clone()
            semi_hard_mask = (neg_dist > hardest_positive_dist.unsqueeze(1)) & \
                           (neg_dist < hardest_positive_dist.unsqueeze(1) + self.margin)
            semi_hard_mask = semi_hard_mask & neg_mask
            
            neg_dist = neg_dist.masked_fill(~semi_hard_mask, float('inf'))
            hardest_negative_dist, _ = neg_dist.min(dim=1)
            
            # If no semi-hard negatives, use hard negatives
            no_semi_hard = torch.isinf(hardest_negative_dist)
            if no_semi_hard.any():
                neg_dist_hard = dist_matrix.clone()
                neg_dist_hard = neg_dist_hard.masked_fill(~neg_mask, float('inf'))
                hard_negative_dist, _ = neg_dist_hard.min(dim=1)
                hardest_negative_dist = torch.where(no_semi_hard, 
                                                   hard_negative_dist, 
                                                   hardest_negative_dist)
        else:
            # Random sampling
            hardest_positive_dist = (dist_matrix * pos_mask.float()).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
            hardest_negative_dist = (dist_matrix * neg_mask.float()).sum(dim=1) / neg_mask.sum(dim=1).clamp(min=1)
        
        # Compute triplet loss
        loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
        loss = loss.mean()
        
        return loss
    
    def _pairwise_distance(self, features):
        """Compute pairwise Euclidean distance"""
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        
        # Compute squared distance
        dot_product = torch.matmul(features, features.t())
        square_norm = dot_product.diag()
        
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        distances = F.relu(distances)  # For numerical stability
        
        # Add small epsilon to avoid sqrt(0)
        mask = (distances == 0).float()
        distances = distances + mask * 1e-16
        distances = torch.sqrt(distances)
        
        # Correct for epsilon
        distances = distances * (1.0 - mask)
        
        return distances


class WeightedCrossEntropyLoss(nn.Module):
    """Cross entropy loss with sample weights"""
    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()
        
    def forward(self, logits, labels, weights=None):
        """
        Args:
            logits: Model predictions (batch_size, num_classes)
            labels: Ground truth labels (batch_size,)
            weights: Sample weights (batch_size,)
        """
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        
        if weights is not None:
            ce_loss = ce_loss * weights
        
        return ce_loss.mean()


class ContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss"""
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, features, labels, centers):
        """
        Args:
            features: Feature embeddings (batch_size, feature_dim)
            labels: Class labels (batch_size,)
            centers: Class centers (num_classes, feature_dim)
        """
        # Normalize features and centers
        features = F.normalize(features, p=2, dim=1)
        centers = F.normalize(centers, p=2, dim=1)
        
        # Compute similarity
        similarity = torch.matmul(features, centers.t()) / self.temperature
        
        # Create positive mask
        batch_size = features.size(0)
        pos_mask = torch.zeros(batch_size, centers.size(0), device=features.device)
        pos_mask[torch.arange(batch_size), labels] = 1
        
        # Compute loss
        exp_sim = torch.exp(similarity)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        loss = -(log_prob * pos_mask).sum(dim=1).mean()
        
        return loss


