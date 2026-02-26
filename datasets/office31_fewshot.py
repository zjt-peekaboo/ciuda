# datasets/office31_fewshot.py
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image, ImageFilter
import numpy as np


class GaussianBlur:
    """Gaussian blur augmentation"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Office31FewShotDataset(Dataset):
    """Office-31 Dataset with few-shot support"""
    def __init__(self, root, domain, transform=None, selected_classes=None):
        self.root = root
        self.domain = domain
        self.transform = transform
        
        self.classes = [
            'back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle',
            'calculator', 'desk_chair', 'desk_lamp', 'desktop_computer', 'file_cabinet',
            'headphones', 'keyboard', 'laptop_computer', 'letter_tray', 'mobile_phone',
            'monitor', 'mouse', 'mug', 'paper_notebook', 'pen',
            'phone', 'printer', 'projector', 'punchers', 'ring_binder',
            'ruler', 'scissors', 'speaker', 'stapler', 'tape_dispenser',
            'trash_can'
        ]
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load data
        self.data = []
        self.labels = []

        domain_path = os.path.join(root, domain)
        
        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            
            if selected_classes is not None and class_idx not in selected_classes:
                continue
            
            class_path = os.path.join(domain_path, class_name)
            if not os.path.exists(class_path):
                continue
            
            for img_name in os.listdir(class_path):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_path, img_name)
                    self.data.append(img_path)
                    self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class TwoCropsTransform:
    """Take two random crops of one image for consistency regularization"""
    def __init__(self, base_transform, strong_transform):
        self.base_transform = base_transform
        self.strong_transform = strong_transform

    def __call__(self, x):
        weak = self.base_transform(x)
        strong = self.strong_transform(x)
        return weak, strong


def get_weak_augmentation():
    """Weak augmentation for labeled data"""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def get_strong_augmentation():
    """Strong augmentation for unlabeled data (FixMatch style)"""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def get_test_transform():
    """Test transform"""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def split_few_shot_data(dataset, n_shot, selected_classes, seed=42):
    """
    Split dataset into labeled (few-shot) and unlabeled sets
    
    Args:
        dataset: Full dataset
        n_shot: Number of labeled samples per class
        selected_classes: List of class indices
        seed: Random seed
    
    Returns:
        labeled_indices, unlabeled_indices
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Group indices by class
    class_indices = {cls: [] for cls in selected_classes}
    for idx, label in enumerate(dataset.labels):
        if label in selected_classes:
            class_indices[label].append(idx)
    
    labeled_indices = []
    unlabeled_indices = []
    
    for cls in selected_classes:
        indices = class_indices[cls]
        random.shuffle(indices)
        
        # Take n_shot samples for labeled set
        n_samples = min(n_shot, len(indices))
        labeled_indices.extend(indices[:n_samples])
        unlabeled_indices.extend(indices[n_samples:])
    
    return labeled_indices, unlabeled_indices


def get_office31_fewshot_loaders(root, domain, n_shot, labeled_batch_size, 
                                 unlabeled_batch_size, num_workers=4,
                                 selected_classes=None, test_classes=None,
                                 strong_aug=True):
    """
    Create few-shot data loaders for Office-31
    
    Args:
        root: Dataset root
        domain: Target domain
        n_shot: Number of labeled samples per class
        labeled_batch_size: Batch size for labeled data
        unlabeled_batch_size: Batch size for unlabeled data
        num_workers: Number of workers
        selected_classes: Classes for current task (to split into labeled/unlabeled)
        test_classes: All seen classes for testing
        strong_aug: Use strong augmentation for unlabeled data
    
    Returns:
        labeled_loader, unlabeled_loader, test_loader
    """
    # Transforms
    weak_transform = get_weak_augmentation()
    strong_transform = get_strong_augmentation() if strong_aug else weak_transform
    test_transform = get_test_transform()
    
    # Create full dataset for current task
    full_dataset = Office31FewShotDataset(
        root=root,
        domain=domain,
        transform=None,
        selected_classes=selected_classes
    )
    
    # Split into labeled and unlabeled
    labeled_indices, unlabeled_indices = split_few_shot_data(
        full_dataset, n_shot, selected_classes
    )
    
    print(f"Few-shot split: {len(labeled_indices)} labeled, {len(unlabeled_indices)} unlabeled")
    
    # Create labeled dataset
    labeled_dataset = Office31FewShotDataset(
        root=root,
        domain=domain,
        transform=weak_transform,
        selected_classes=selected_classes
    )
    labeled_subset = Subset(labeled_dataset, labeled_indices)
    
    # Create unlabeled dataset with two-crop transform
    if strong_aug:
        unlabeled_transform = TwoCropsTransform(weak_transform, strong_transform)
    else:
        unlabeled_transform = weak_transform
    
    unlabeled_dataset = Office31FewShotDataset(
        root=root,
        domain=domain,
        transform=unlabeled_transform,
        selected_classes=selected_classes
    )
    unlabeled_subset = Subset(unlabeled_dataset, unlabeled_indices)
    
    # Create test dataset
    test_dataset = Office31FewShotDataset(
        root=root,
        domain=domain,
        transform=test_transform,
        selected_classes=test_classes
    )
    
    # Create data loaders
    labeled_loader = DataLoader(
        labeled_subset,
        batch_size=labeled_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    unlabeled_loader = None
    if len(unlabeled_indices) > 0:
        unlabeled_loader = DataLoader(
            unlabeled_subset,
            batch_size=unlabeled_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=labeled_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return labeled_loader, unlabeled_loader, test_loader