# models/resnet.py
import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50Backbone(nn.Module):
    """ResNet50 feature extractor"""
    def __init__(self, pretrained=True):
        super(ResNet50Backbone, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)
        # Remove the final classification layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 2048
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


class Classifier(nn.Module):
    """Classification head"""
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.fc(x)




