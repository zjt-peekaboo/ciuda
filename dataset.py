# dataset.py
"""Office-31 dataset loading and incremental task splitting."""
import os
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from loguru import logger


# Office-31 domain folder names
DOMAIN_FOLDERS = {
    "amazon": "amazon",
    "dslr": "dslr",
    "webcam": "webcam",
}


class Office31Dataset(Dataset):
    """Office-31 dataset loader."""

    def __init__(
        self,
        root: str,
        domain: str,
        transform=None,
        class_indices: Optional[List[int]] = None,
        class_mapping: Optional[Dict[int, int]] = None,
    ):
        """
        Args:
            root: path to office31 root (contains amazon/, dslr/, webcam/)
            domain: one of 'amazon', 'dslr', 'webcam'
            transform: image transforms
            class_indices: if given, only load these original class indices
            class_mapping: maps original class idx -> global sequential idx
        """
        self.root = root
        self.domain = domain
        self.transform = transform
        self.class_mapping = class_mapping

        domain_path = os.path.join(root, DOMAIN_FOLDERS[domain], "images")
        if not os.path.isdir(domain_path):
            # try without /images
            domain_path = os.path.join(root, DOMAIN_FOLDERS[domain])

        # Discover classes (sorted for reproducibility)
        all_classes = sorted(
            [d for d in os.listdir(domain_path) if os.path.isdir(os.path.join(domain_path, d))]
        )
        self.class_to_idx = {c: i for i, c in enumerate(all_classes)}

        # Build sample list
        self.samples: List[Tuple[str, int]] = []
        for cls_name in all_classes:
            cls_idx = self.class_to_idx[cls_name]
            if class_indices is not None and cls_idx not in class_indices:
                continue
            cls_dir = os.path.join(domain_path, cls_name)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    self.samples.append((os.path.join(cls_dir, fname), cls_idx))

        logger.info(
            f"Office31 [{domain}]: {len(self.samples)} images, "
            f"{len(set(s[1] for s in self.samples))} classes loaded"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        # Apply class mapping if provided
        if self.class_mapping is not None:
            label = self.class_mapping[label]
        return img, label, idx


class TaskSplitter:
    """Split 31 classes into disjoint incremental tasks."""

    def __init__(self, num_classes: int, task_sizes: List[int], seed: int):
        assert sum(task_sizes) == num_classes, (
            f"Task sizes {task_sizes} must sum to {num_classes}"
        )
        self.num_classes = num_classes
        self.task_sizes = task_sizes
        self.seed = seed

        # Shuffle class indices
        rng = random.Random(seed)
        self.class_order = list(range(num_classes))
        rng.shuffle(self.class_order)

        # Build task -> class indices mapping
        self.task_classes: List[List[int]] = []
        offset = 0
        for size in task_sizes:
            self.task_classes.append(self.class_order[offset: offset + size])
            offset += size

        # Global mapping: original class idx -> sequential label (0..30)
        self.class_mapping: Dict[int, int] = {}
        for new_idx, orig_idx in enumerate(self.class_order):
            self.class_mapping[orig_idx] = new_idx

        logger.info(f"Task split (seed={seed}): {[len(t) for t in self.task_classes]}")
        for i, tc in enumerate(self.task_classes):
            logger.info(f"  Task {i+1}: original classes {tc} -> mapped {[self.class_mapping[c] for c in tc]}")

    def get_classes_up_to_task(self, task_id: int) -> List[int]:
        """Return all original class indices seen up to and including task_id (0-based)."""
        classes = []
        for t in range(task_id + 1):
            classes.extend(self.task_classes[t])
        return classes

    def get_mapped_classes_for_task(self, task_id: int) -> List[int]:
        """Return mapped (sequential) class indices for a specific task."""
        return [self.class_mapping[c] for c in self.task_classes[task_id]]

    def get_all_mapped_classes_up_to(self, task_id: int) -> List[int]:
        """Return all mapped class indices seen up to task_id."""
        return [self.class_mapping[c] for c in self.get_classes_up_to_task(task_id)]


def get_transforms(img_size: int = 224, mode: str = "train"):
    """Get image transforms."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    elif mode == "weak":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            normalize,
        ])
    elif mode == "strong":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:  # test / eval
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ])


class MultiViewDataset(Dataset):
    """Wraps a base dataset to return multiple augmented views per sample."""

    def __init__(self, base_dataset: Office31Dataset, img_size: int = 224):
        self.base = base_dataset
        self.weak_transform = get_transforms(img_size, "weak")
        self.strong_transform = get_transforms(img_size, "strong")
        # We'll also need the raw PIL for multi-view
        self.to_tensor_transform = get_transforms(img_size, "test")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        path, label = self.base.samples[idx]
        img = Image.open(path).convert("RGB")

        view_weak = self.weak_transform(img)
        view_strong = self.strong_transform(img)
        view_test = self.to_tensor_transform(img)

        mapped_label = label
        if self.base.class_mapping is not None:
            mapped_label = self.base.class_mapping[label]

        return {
            "weak": view_weak,
            "strong": view_strong,
            "test": view_test,
            "label": mapped_label,
            "orig_label": label,
            "idx": idx,
        }
