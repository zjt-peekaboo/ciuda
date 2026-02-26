# utils/prototype.py
import torch
import torch.nn.functional as F
import numpy as np


class PrototypeManager:
    """
    Manage class prototypes for few-shot learning
    Uses exponential moving average for stable prototype updates
    """
    def __init__(self, feature_dim, num_classes, momentum=0.9, device='cuda'):
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.momentum = momentum
        self.device = device
        
        # Initialize prototypes
        self.prototypes = torch.zeros(num_classes, feature_dim).to(device)
        self.prototype_counts = torch.zeros(num_classes).to(device)
        self.initialized = torch.zeros(num_classes, dtype=torch.bool).to(device)
        
    def initialize_prototypes(self, features, labels):
        """
        Initialize prototypes from few-shot labeled samples
        
        Args:
            features: Feature embeddings (N, feature_dim)
            labels: Class labels (N,)
        """
        features = F.normalize(features, p=2, dim=1)
        
        unique_labels = torch.unique(labels)
        for label in unique_labels:
            label = label.item()
            mask = labels == label
            class_features = features[mask]
            
            # Compute class centroid
            centroid = class_features.mean(dim=0)
            centroid = F.normalize(centroid.unsqueeze(0), p=2, dim=1).squeeze(0)
            
            self.prototypes[label] = centroid
            self.prototype_counts[label] = mask.sum()
            self.initialized[label] = True
    
    def update_prototypes(self, features, labels, use_momentum=True):
        """
        Update prototypes with new samples
        
        Args:
            features: Feature embeddings (N, feature_dim)
            labels: Class labels (N,)
            use_momentum: Use exponential moving average
        """
        features = F.normalize(features, p=2, dim=1)
        
        unique_labels = torch.unique(labels)
        for label in unique_labels:
            label = label.item()
            mask = labels == label
            class_features = features[mask]
            
            # Compute new centroid
            new_centroid = class_features.mean(dim=0)
            new_centroid = F.normalize(new_centroid.unsqueeze(0), p=2, dim=1).squeeze(0)
            
            if self.initialized[label] and use_momentum:
                # EMA update
                self.prototypes[label] = (
                    self.momentum * self.prototypes[label] + 
                    (1 - self.momentum) * new_centroid
                )
                self.prototypes[label] = F.normalize(
                    self.prototypes[label].unsqueeze(0), p=2, dim=1
                ).squeeze(0)
            else:
                # Direct update
                self.prototypes[label] = new_centroid
                self.initialized[label] = True
            
            self.prototype_counts[label] += mask.sum()
    
    def get_prototypes(self, class_indices=None):
        """
        Get prototypes for specified classes
        
        Args:
            class_indices: List of class indices (default: all initialized)
        
        Returns:
            prototypes: (num_classes, feature_dim)
        """
        if class_indices is None:
            mask = self.initialized
            return self.prototypes[mask]
        else:
            return self.prototypes[class_indices]
    
    def predict_by_prototype(self, features, class_indices):
        """
        Predict labels based on nearest prototype
        
        Args:
            features: Feature embeddings (N, feature_dim)
            class_indices: List of valid class indices
        
        Returns:
            predictions: Predicted class indices
            distances: Distance to nearest prototype
        """
        features = F.normalize(features, p=2, dim=1)
        prototypes = self.prototypes[class_indices]
        prototypes = F.normalize(prototypes, p=2, dim=1)
        
        # Compute similarity (negative distance)
        similarity = torch.matmul(features, prototypes.t())
        
        # Find nearest prototype
        max_sim, pred_idx = similarity.max(dim=1)
        predictions = torch.tensor(class_indices, device=features.device)[pred_idx]
        distances = 1 - max_sim  # Convert similarity to distance
        
        return predictions, distances
    
    def compute_prototype_confidence(self, features, class_indices, temperature=0.1):
        """
        Compute confidence based on prototype similarity
        
        Args:
            features: Feature embeddings (N, feature_dim)
            class_indices: List of valid class indices
            temperature: Temperature for softmax
        
        Returns:
            confidence: Confidence scores (N,)
            predictions: Predicted class indices (N,)
        """
        features = F.normalize(features, p=2, dim=1)
        prototypes = self.prototypes[class_indices]
        prototypes = F.normalize(prototypes, p=2, dim=1)
        
        # Compute similarity
        similarity = torch.matmul(features, prototypes.t()) / temperature
        
        # Softmax to get probabilities
        probs = F.softmax(similarity, dim=1)
        
        # Max probability as confidence
        confidence, pred_idx = probs.max(dim=1)
        predictions = torch.tensor(class_indices, device=features.device)[pred_idx]
        
        return confidence, predictions
    
    def get_prototype_loss(self, features, labels, class_indices):
        """
        Compute prototype alignment loss (encourage features to be close to prototypes)
        
        Args:
            features: Feature embeddings (N, feature_dim)
            labels: Ground truth labels (N,)
            class_indices: List of valid class indices
        
        Returns:
            loss: Prototype alignment loss
        """
        features = F.normalize(features, p=2, dim=1)
        
        # Get target prototypes
        target_prototypes = self.prototypes[labels]
        target_prototypes = F.normalize(target_prototypes, p=2, dim=1)
        
        # Compute distance loss
        distances = 1 - (features * target_prototypes).sum(dim=1)
        loss = distances.mean()
        
        return loss
    
    def get_inter_prototype_distance(self, class_indices):
        """
        Compute minimum distance between prototypes (for regularization)
        
        Args:
            class_indices: List of class indices
        
        Returns:
            min_distance: Minimum pairwise distance
        """
        prototypes = self.prototypes[class_indices]
        prototypes = F.normalize(prototypes, p=2, dim=1)
        
        # Compute pairwise distances
        similarity = torch.matmul(prototypes, prototypes.t())
        
        # Mask out diagonal
        mask = torch.eye(len(class_indices), device=self.device).bool()
        similarity.masked_fill_(mask, -float('inf'))
        
        # Get maximum similarity (minimum distance)
        max_sim = similarity.max()
        min_distance = 1 - max_sim
        
        return min_distance
    
    def refine_prototypes_with_kmeans(self, features, labels, class_indices, n_iter=10):
        """
        Refine prototypes using k-means style updates
        
        Args:
            features: Feature embeddings from all samples
            labels: Labels
            class_indices: Classes to refine
            n_iter: Number of iterations
        """
        features = F.normalize(features, p=2, dim=1)
        
        for _ in range(n_iter):
            for cls in class_indices:
                # Find samples assigned to this class
                distances = torch.norm(
                    features - self.prototypes[cls].unsqueeze(0), 
                    p=2, dim=1
                )
                
                # Get samples within threshold
                threshold = distances.median()
                mask = (distances < threshold) & (labels == cls)
                
                if mask.sum() > 0:
                    # Update prototype
                    new_proto = features[mask].mean(dim=0)
                    new_proto = F.normalize(new_proto.unsqueeze(0), p=2, dim=1).squeeze(0)
                    self.prototypes[cls] = new_proto
    
    def save_prototypes(self, path):
        """Save prototypes to file"""
        torch.save({
            'prototypes': self.prototypes.cpu(),
            'prototype_counts': self.prototype_counts.cpu(),
            'initialized': self.initialized.cpu()
        }, path)
    
    def load_prototypes(self, path):
        """Load prototypes from file"""
        checkpoint = torch.load(path)
        self.prototypes = checkpoint['prototypes'].to(self.device)
        self.prototype_counts = checkpoint['prototype_counts'].to(self.device)
        self.initialized = checkpoint['initialized'].to(self.device)
