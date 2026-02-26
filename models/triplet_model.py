# models/triplet_model.py
import torch
import torch.nn as nn


class TripletModel(nn.Module):
    """Complete model with backbone and classifier"""

    def __init__(self, backbone, classifier):
        super(TripletModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x, return_features=False):
        features = self.backbone(x)
        logits = self.classifier(features)

        if return_features:
            return logits, features
        return logits

    def get_features(self, x):
        """Extract features only"""
        return self.backbone(x)

    def get_classifier_weights(self):
        """Get classifier weights (class centers)"""
        return self.classifier.fc.weight.data