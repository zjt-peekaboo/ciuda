# main.py
"""
E3P-SFCIDA: Energy-Enhanced Prototype Propagation for
Source-Free Class Incremental Domain Adaptation on Office-31.

Usage:
    python main.py --source amazon --target webcam
    python main.py --source dslr --target amazon --seed 123
"""
import os
import sys
import argparse
import time

import torch
from loguru import logger

from config import E3PConfig
from model import E3PModel
from dataset import (
    Office31Dataset,
    TaskSplitter,
    get_transforms,
)
from trainer import SourceTrainer, TargetTrainer
from utils import set_seed, check_office31_structure


def setup_logger(output_dir: str):
    """Configure loguru logger."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO",
    )
    log_path = os.path.join(output_dir, "e3p_run.log")
    logger.add(log_path, rotation="50 MB", level="DEBUG")
    logger.info(f"Log file: {log_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="E3P-SFCIDA on Office-31")
    parser.add_argument("--data_root", type=str, default="./data/office31",
                        help="Path to Office-31 dataset root")
    parser.add_argument("--source", type=str, default="amazon",
                        choices=["amazon", "dslr", "webcam"])
    parser.add_argument("--target", type=str, default="webcam",
                        choices=["amazon", "dslr", "webcam"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--source_epochs", type=int, default=30)
    parser.add_argument("--target_epochs", type=int, default=20)
    parser.add_argument("--buffer_capacity", type=int, default=300)
    parser.add_argument("--source_lr", type=float, default=0.001)
    parser.add_argument("--target_lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Default batch size for both source and target")
    parser.add_argument("--source_batch_size", type=int, default=None,
                        help="Source domain batch size (overrides --batch_size)")
    parser.add_argument("--target_batch_size", type=int, default=None,
                        help="Target domain batch size (overrides --batch_size)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_source", action="store_true",
                        help="Skip source training, load existing model")
    # Additional hyperparameters for tuning
    parser.add_argument("--proto_temp", type=float, default=None,
                        help="Prototype temperature (default: 0.07)")
    parser.add_argument("--proto_ema", type=float, default=None,
                        help="Prototype EMA beta (default: 0.9)")
    parser.add_argument("--reliability_threshold", type=float, default=None,
                        help="Reliability threshold for prototype update (default: 0.3)")
    parser.add_argument("--class_detect_alpha", type=int, default=None,
                        help="Class discovery voting threshold (default: 3)")
    parser.add_argument("--lambda_pl", type=float, default=None,
                        help="Pseudo-label loss weight (default: 1.0)")
    parser.add_argument("--lambda_proto", type=float, default=None,
                        help="Prototype loss weight (default: 0.1)")
    parser.add_argument("--lambda_dist", type=float, default=None,
                        help="Distillation loss weight (default: 1.0)")
    # Triplet loss for source training
    parser.add_argument("--use_triplet", action="store_true",
                        help="Use triplet loss in source training")
    parser.add_argument("--no_triplet", action="store_true",
                        help="Disable triplet loss in source training")
    parser.add_argument("--triplet_margin", type=float, default=None,
                        help="Triplet loss margin (default: 0.5)")
    parser.add_argument("--lambda_triplet", type=float, default=None,
                        help="Triplet loss weight (default: 0.1)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Build config
    config = E3PConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        source_domain=args.source,
        target_domain=args.target,
        seed=args.seed,
        source_epochs=args.source_epochs,
        target_epochs_per_task=args.target_epochs,
        buffer_total_capacity=args.buffer_capacity,
        source_lr=args.source_lr,
        target_lr=args.target_lr,
        source_batch_size=args.source_batch_size if args.source_batch_size else args.batch_size,
        target_batch_size=args.target_batch_size if args.target_batch_size else args.batch_size,
        device=args.device if torch.cuda.is_available() else "cpu",
        proto_temperature=args.proto_temp if args.proto_temp is not None else 0.07,
        proto_ema_beta=args.proto_ema if args.proto_ema is not None else 0.9,
        reliability_threshold_gamma=args.reliability_threshold if args.reliability_threshold is not None else 0.3,
        class_detect_alpha=args.class_detect_alpha if args.class_detect_alpha is not None else 3,
        lambda_pl=args.lambda_pl if args.lambda_pl is not None else 1.0,
        lambda_proto=args.lambda_proto if args.lambda_proto is not None else 0.1,
        lambda_dist=args.lambda_dist if args.lambda_dist is not None else 1.0,
        # Triplet loss settings
        use_triplet=not args.no_triplet,  # default True, use --no_triplet to disable
        triplet_margin=args.triplet_margin if args.triplet_margin is not None else 0.5,
        lambda_triplet=args.lambda_triplet if args.lambda_triplet is not None else 0.1,
    )
    config.source_model_path = os.path.join(
        config.output_dir, f"source_{config.source_domain}.pth"
    )

    os.makedirs(config.output_dir, exist_ok=True)
    setup_logger(config.output_dir)

    logger.info("=" * 60)
    logger.info("E3P-SFCIDA: Energy-Enhanced Prototype Propagation")
    logger.info("=" * 60)
    logger.info(f"Source: {config.source_domain} -> Target: {config.target_domain}")
    logger.info(f"Task sizes: {config.task_sizes} (total {config.num_classes} classes)")
    logger.info(f"Device: {config.device}")
    logger.info(f"Seed: {config.seed}")

    set_seed(config.seed)

    # Verify dataset
    if not check_office31_structure(config.data_root):
        logger.error(
            "Office-31 dataset not found. Please download and extract to: "
            f"{config.data_root}\n"
            "Expected structure:\n"
            "  office31/\n"
            "    amazon/images/{back_pack, bike, ...}/\n"
            "    dslr/images/{back_pack, bike, ...}/\n"
            "    webcam/images/{back_pack, bike, ...}/\n"
        )
        sys.exit(1)

    # Task splitter: [10, 10, 10, 1]
    splitter = TaskSplitter(
        num_classes=config.num_classes,
        task_sizes=config.task_sizes,
        seed=config.seed,
    )

    # ========== Stage 1: Source Domain Training ==========
    source_model = E3PModel(
        num_classes=config.num_classes,
        feature_dim=2048,
        adapter_dim=config.adapter_dim,
        pretrained=config.pretrained_imagenet,
    )

    if args.skip_source and os.path.exists(config.source_model_path):
        logger.info(f"Loading existing source model from {config.source_model_path}")
        source_model.load_state_dict(
            torch.load(config.source_model_path, map_location=config.device)
        )
        source_model.to(config.device)
    else:
        # Source dataset: all 31 classes, labeled
        source_dataset = Office31Dataset(
            root=config.data_root,
            domain=config.source_domain,
            transform=get_transforms(config.img_size, "train"),
            class_indices=None,  # all classes
            class_mapping=splitter.class_mapping,
        )

        source_trainer = SourceTrainer(config)
        source_model = source_trainer.train(source_model, source_dataset)

    # Evaluate source model on source domain (sanity check)
    logger.info("Source model sanity check on source domain:")
    source_eval_dataset = Office31Dataset(
        root=config.data_root,
        domain=config.source_domain,
        transform=get_transforms(config.img_size, "test"),
        class_indices=None,
        class_mapping=splitter.class_mapping,
    )
    _eval_source(source_model, source_eval_dataset, config)

    # ========== Stage 2: Target Incremental Adaptation ==========
    # Full target dataset for evaluation (all 31 classes)
    target_eval_dataset = Office31Dataset(
        root=config.data_root,
        domain=config.target_domain,
        transform=get_transforms(config.img_size, "test"),
        class_indices=None,
        class_mapping=splitter.class_mapping,
    )

    target_trainer = TargetTrainer(config, splitter)

    # Full target base dataset (used to create per-task subsets inside trainer)
    target_base_dataset = Office31Dataset(
        root=config.data_root,
        domain=config.target_domain,
        transform=get_transforms(config.img_size, "train"),
        class_indices=None,
        class_mapping=splitter.class_mapping,
    )

    start_time = time.time()
    final_model = target_trainer.adapt(
        source_model=source_model,
        target_base_dataset=target_base_dataset,
        eval_dataset=target_eval_dataset,
    )
    elapsed = time.time() - start_time
    logger.info(f"Total adaptation time: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # Save final model
    final_path = os.path.join(
        config.output_dir,
        f"e3p_final_{config.source_domain}_{config.target_domain}.pth",
    )
    torch.save(final_model.state_dict(), final_path)
    logger.info(f"Final model saved to {final_path}")


@torch.no_grad()
def _eval_source(model: E3PModel, dataset: Office31Dataset, config: E3PConfig):
    """Quick evaluation of source model on source domain."""
    device = torch.device(config.device)
    model.eval()
    model.to(device)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.source_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    correct = 0
    total = 0
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        logits, _ = model(images, use_adapter=False)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    acc = 100.0 * correct / max(total, 1)
    logger.info(f"  Source domain accuracy: {acc:.2f}% ({correct}/{total})")


if __name__ == "__main__":
    main()
