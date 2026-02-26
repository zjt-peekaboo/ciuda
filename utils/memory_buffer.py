# utils/memory_buffer.py
import torch
import numpy as np


class MemoryBuffer:
    """Memory buffer for storing exemplars"""

    def __init__(self, max_size, exemplars_per_class):
        self.max_size = max_size
        self.exemplars_per_class = exemplars_per_class
        self.buffer = {}  # {class_id: {'images': [], 'features': []}}

    def add_class(self, class_id, images, features, confidences):
        """
        Add exemplars for a new class

        Args:
            class_id: Class identifier
            images: Image tensors (n_samples, C, H, W)
            features: Feature embeddings (n_samples, feature_dim)
            confidences: Sample confidence scores (n_samples,)
        """
        n_samples = min(len(images), self.exemplars_per_class)

        # Sort by confidence
        sorted_indices = torch.argsort(confidences, descending=True)
        selected_indices = sorted_indices[:n_samples]

        # Store exemplars
        self.buffer[class_id] = {
            'images': images[selected_indices].cpu(),
            'features': features[selected_indices].cpu()
        }

    def get_class_data(self, class_id):
        """Get stored data for a class"""
        if class_id in self.buffer:
            return self.buffer[class_id]['images'], self.buffer[class_id]['features']
        return None, None

    def get_all_data(self):
        """Get all stored data"""
        all_images = []
        all_labels = []

        for class_id, data in self.buffer.items():
            images = data['images']
            all_images.append(images)
            all_labels.extend([class_id] * len(images))

        if len(all_images) == 0:
            return None, None

        all_images = torch.cat(all_images, dim=0)
        all_labels = torch.tensor(all_labels)

        return all_images, all_labels

    def update_class(self, class_id, images, features, confidences):
        """Update exemplars for an existing class"""
        if class_id in self.buffer:
            self.add_class(class_id, images, features, confidences)

    def get_num_classes(self):
        """Get number of classes in buffer"""
        return len(self.buffer)

    def get_size(self):
        """Get total number of samples in buffer"""
        total = 0
        for data in self.buffer.values():
            total += len(data['images'])
        return total
