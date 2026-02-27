# config.py
"""E3P-SFCIDA Configuration."""
from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class E3PConfig:
    # --- Paths ---
    data_root: str = "./data/office31"
    output_dir: str = "./output"
    source_model_path: str = "./output/source_model.pth"

    # --- Dataset ---
    source_domain: str = "amazon"
    target_domain: str = "webcam"
    num_classes: int = 31
    # Task split: [5, 5, 5, 5, 5, 6] - 6 tasks
    task_sizes: List[int] = field(default_factory=lambda: [5, 5, 5, 5, 5, 6])
    seed: int = 42
    img_size: int = 224

    # --- Model ---
    backbone: str = "resnet50"
    pretrained_imagenet: bool = True
    feature_dim: int = 2048
    adapter_dim: int = 64  # bottleneck dim for adapter
    freeze_backbone: bool = True  # freeze most of backbone

    # --- Source Training ---
    source_epochs: int = 30
    source_lr: float = 0.001
    source_batch_size: int = 64

    # --- Target Adaptation ---
    target_epochs_per_task: int = 20
    target_lr: float = 0.001
    target_batch_size: int = 64

    # --- E3P Hyperparameters ---
    # Reliability
    num_augment_views: int = 4  # weak, strong, dropout1, dropout2
    energy_temperature: float = 1.0

    # Prototype
    proto_ema_beta: float = 0.9
    proto_temperature: float = 0.07
    reliability_threshold_gamma: float = 0.3

    # Buffer
    buffer_total_capacity: int = 300  # total images in buffer
    buffer_sweet_spot_m: float = 0.65  # target cosine sim for contribution

    # Loss weights
    lambda_pl: float = 1.0
    lambda_proto: float = 0.1
    lambda_dist: float = 1.0

    # Class detection threshold (lowered for 5 classes per task)
    class_detect_alpha: int = 3

    # --- Misc ---
    num_workers: int = 4
    device: str = "cuda"
    log_interval: int = 10

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
