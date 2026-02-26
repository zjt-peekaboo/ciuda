# datasets/office31.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class Office31Dataset(Dataset):
    """Office-31 Dataset"""

    def __init__(self, root, domain, train=True, transform=None, selected_classes=None):
        """
        Args:
            root: Root directory of Office-31 dataset
            domain: One of 'amazon', 'dslr', 'webcam'
            train: Whether to load training or test set
            transform: Image transformations
            selected_classes: List of class indices to include (for incremental learning)
        """
        self.root = root
        self.domain = domain
        self.train = train
        self.transform = transform

        # Office-31 class names
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

            # Skip if class not selected
            if selected_classes is not None and class_idx not in selected_classes:
                continue

            class_path = os.path.join(domain_path, class_name)
            print(class_path)

            if not os.path.exists(class_path):
                continue

            for img_name in os.listdir(class_path):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_path, img_name)
                    self.data.append(img_path)
                    self.labels.append(class_idx)

        print(f"Loaded {len(self.data)} images from {domain} domain")
        if selected_classes:
            print(f"Selected classes: {selected_classes}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label


def get_office31_loaders(root, domain, batch_size, num_workers=4,
                         train=True, selected_classes=None):
    """
    Create data loaders for Office-31 dataset

    Args:
        root: Root directory of Office-31 dataset
        domain: One of 'amazon', 'dslr', 'webcam'
        batch_size: Batch size
        num_workers: Number of workers for data loading
        train: Whether to load training or test set
        selected_classes: List of class indices to include

    Returns:
        DataLoader
    """
    # Define transforms
    if train:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    # Create dataset
    dataset = Office31Dataset(
        root=root,
        domain=domain,
        train=train,
        transform=transform,
        selected_classes=selected_classes
    )

    # Create data loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train
    )

    return loader