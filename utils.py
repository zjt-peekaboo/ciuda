# utils.py
"""Utility functions for E3P-SFCIDA."""
import os
import random
import numpy as np
import torch
from loguru import logger


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def check_office31_structure(data_root: str) -> bool:
    """Check if Office-31 dataset is properly structured."""
    required_domains = ["amazon", "dslr", "webcam"]
    for domain in required_domains:
        domain_path = os.path.join(data_root, domain, "images")
        if not os.path.isdir(domain_path):
            domain_path = os.path.join(data_root, domain)
            if not os.path.isdir(domain_path):
                logger.error(
                    f"Domain folder not found: {domain_path}. "
                    f"Expected structure: {data_root}/{{amazon,dslr,webcam}}/images/{{class_folders}}"
                )
                return False
        # Check at least some class folders exist
        subfolders = [
            d for d in os.listdir(domain_path)
            if os.path.isdir(os.path.join(domain_path, d))
        ]
        if len(subfolders) < 31:
            logger.warning(
                f"Domain '{domain}' has {len(subfolders)} class folders (expected 31)"
            )
    logger.info(f"Office-31 dataset structure verified at {data_root}")
    return True


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ForgettingMeasure:
    """
    Track per-task accuracy over time to compute forgetting.
    FM_j = max_{t<=T-1} acc_t(j) - acc_T(j)
    """

    def __init__(self):
        # task_eval_id -> {eval_task -> accuracy}
        self.history: dict = {}

    def record(self, current_task: int, eval_task: int, accuracy: float):
        if current_task not in self.history:
            self.history[current_task] = {}
        self.history[current_task][eval_task] = accuracy

    def compute_forgetting(self) -> float:
        """Compute average forgetting across all tasks except the last."""
        if len(self.history) < 2:
            return 0.0

        tasks = sorted(self.history.keys())
        final_task = tasks[-1]
        forgetting_values = []

        for eval_task in tasks[:-1]:
            # Find peak accuracy for this eval_task across all time steps
            peak = max(
                self.history[t].get(eval_task, 0.0)
                for t in tasks
                if eval_task in self.history.get(t, {})
            )
            final_acc = self.history[final_task].get(eval_task, 0.0)
            forgetting_values.append(peak - final_acc)

        return sum(forgetting_values) / max(len(forgetting_values), 1)

    def summary(self) -> str:
        lines = ["Forgetting Measure History:"]
        for t in sorted(self.history.keys()):
            accs = self.history[t]
            acc_str = ", ".join(f"T{k}={v:.1f}%" for k, v in sorted(accs.items()))
            lines.append(f"  After Task {t}: {acc_str}")
        lines.append(f"  Average Forgetting: {self.compute_forgetting():.2f}%")
        return "\n".join(lines)
