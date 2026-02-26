"""
Few-Shot CIDA: Few-Shot Class Incremental Domain Adaptation
Main training script with few-shot labeled samples in target domain
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime

from models.resnet import ResNet50Backbone, Classifier
from models.triplet_model import TripletModel
from datasets.office31_fewshot import get_office31_fewshot_loaders
from utils.losses import TripletLoss, WeightedCrossEntropyLoss, ContrastiveLoss
from utils.confidence import ConfidenceEstimator
from utils.memory_buffer import MemoryBuffer
from utils.metrics import AverageAccuracy, ForgettingMeasure
from utils.prototype import PrototypeManager
from trainers.source_trainer import SourceTrainer
from trainers.target_trainer_fewshot import FewShotTargetTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Few-Shot CIDA Training')
    
    # Dataset settings
    parser.add_argument('--data_root', type=str, default='./data/office31',
                        help='Path to Office-31 dataset')
    parser.add_argument('--source', type=str, default='amazon',
                        choices=['amazon', 'dslr', 'webcam'])
    parser.add_argument('--target', type=str, default='webcam',
                        choices=['amazon', 'dslr', 'webcam'])
    parser.add_argument('--num_classes_per_task', type=int, default=5,
                        help='Number of classes per incremental task')
    
    # Few-shot settings
    parser.add_argument('--n_shot', type=int, default=5,
                        help='Number of labeled samples per class (few-shot)')
    parser.add_argument('--use_unlabeled', action='store_true', default=True,
                        help='Use unlabeled target samples')
    
    # Model settings
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--pretrained', action='store_true', default=True)
    
    # Training settings - Source
    parser.add_argument('--source_epochs', type=int, default=50)
    parser.add_argument('--source_lr', type=float, default=0.001)
    parser.add_argument('--source_batch_size', type=int, default=32)
    
    # Training settings - Target
    parser.add_argument('--target_epochs', type=int, default=50,
                        help='Increased epochs for few-shot learning')
    parser.add_argument('--target_lr', type=float, default=0.0005,
                        help='Lower LR for fine-tuning')
    parser.add_argument('--target_batch_size', type=int, default=16)
    parser.add_argument('--unlabeled_batch_size', type=int, default=32)
    
    # Triplet loss settings
    parser.add_argument('--triplet_margin', type=float, default=0.5)
    parser.add_argument('--alpha_triplet', type=float, default=0.3)
    
    # Few-shot specific loss weights
    parser.add_argument('--beta_supervised', type=float, default=2.0,
                        help='Weight for supervised loss on labeled data')
    parser.add_argument('--beta_prototype', type=float, default=1.0,
                        help='Weight for prototype loss')
    parser.add_argument('--beta_pseudo', type=float, default=0.5,
                        help='Weight for pseudo-label loss')
    parser.add_argument('--beta_consistency', type=float, default=0.3,
                        help='Weight for consistency regularization')
    parser.add_argument('--beta_triplet', type=float, default=0.2)
    parser.add_argument('--beta_replay', type=float, default=1.0)
    
    # Pseudo-labeling settings
    parser.add_argument('--confidence_threshold', type=float, default=0.95,
                        help='Confidence threshold for pseudo-labeling')
    parser.add_argument('--proto_confidence_threshold', type=float, default=0.8,
                        help='Prototype-based confidence threshold')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Warmup epochs before using pseudo-labels')
    
    # Data augmentation settings
    parser.add_argument('--strong_aug', action='store_true', default=True,
                        help='Use strong augmentation for consistency')
    parser.add_argument('--uda_steps', type=int, default=2,
                        help='Number of UDA steps per iteration')
    
    # Memory buffer settings
    parser.add_argument('--memory_size', type=int, default=2000)
    parser.add_argument('--exemplars_per_class', type=int, default=20)
    parser.add_argument('--prioritize_labeled', action='store_true', default=True,
                        help='Prioritize labeled samples in memory')
    
    # Prototype settings
    parser.add_argument('--update_prototypes', action='store_true', default=True,
                        help='Update prototypes during training')
    parser.add_argument('--prototype_momentum', type=float, default=0.9,
                        help='Momentum for prototype update')
    
    # Other settings
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='both',
                        choices=['source', 'target', 'both'])
    parser.add_argument('--source_model_path', type=str, default=None)
    
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_directories(args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"fewshot_{args.n_shot}shot_{args.source}_to_{args.target}_{timestamp}"
    
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, exp_name)
    args.log_dir = os.path.join(args.log_dir, exp_name)
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    return exp_name


def main():
    args = parse_args()
    set_seed(args.seed)
    exp_name = setup_directories(args)
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Experiment: {exp_name}")
    print(f"Source: {args.source} -> Target: {args.target}")
    print(f"Few-shot setting: {args.n_shot}-shot per class")
    
    total_classes = 31
    num_tasks = total_classes // args.num_classes_per_task
    print(f"Total classes: {total_classes}, Tasks: {num_tasks}")
    
    # ==================== SOURCE DOMAIN TRAINING ====================
    if args.mode in ['source', 'both']:
        print("\n" + "="*60)
        print("STAGE 1: SOURCE DOMAIN TRAINING")
        print("="*60)
        
        from datasets.office31 import get_office31_loaders
        source_loader = get_office31_loaders(
            root=args.data_root,
            domain=args.source,
            batch_size=args.source_batch_size,
            num_workers=args.num_workers,
            train=True
        )
        
        backbone = ResNet50Backbone(pretrained=args.pretrained).to(device)
        classifier = Classifier(input_dim=2048, num_classes=total_classes).to(device)
        triplet_model = TripletModel(backbone, classifier).to(device)
        
        source_trainer = SourceTrainer(model=triplet_model, device=device, args=args)
        source_trainer.train(source_loader, args.source_epochs)
        
        source_model_path = os.path.join(args.checkpoint_dir, 'source_model_final.pth')
        source_trainer.save_checkpoint(source_model_path)
        print(f"Source model saved to: {source_model_path}")
        
        args.source_model_path = source_model_path
    
    # ==================== TARGET DOMAIN FEW-SHOT ADAPTATION ====================
    if args.mode in ['target', 'both']:
        print("\n" + "="*60)
        print("STAGE 2: FEW-SHOT TARGET DOMAIN ADAPTATION")
        print("="*60)
        
        if args.source_model_path is None:
            raise ValueError("Source model path is required!")
        
        print(f"Loading source model from: {args.source_model_path}")
        
        backbone = ResNet50Backbone(pretrained=False).to(device)
        classifier = Classifier(input_dim=2048, num_classes=total_classes).to(device)
        target_model = TripletModel(backbone, classifier).to(device)
        
        checkpoint = torch.load(args.source_model_path, map_location=device)
        target_model.load_state_dict(checkpoint['model_state_dict'])
        print("Source model loaded successfully!")
        
        # Initialize components
        confidence_estimator = ConfidenceEstimator(model=target_model, device=device)
        memory_buffer = MemoryBuffer(
            max_size=args.memory_size,
            exemplars_per_class=args.exemplars_per_class
        )
        prototype_manager = PrototypeManager(
            feature_dim=2048,
            num_classes=total_classes,
            momentum=args.prototype_momentum,
            device=device
        )
        
        # Initialize metrics
        avg_accuracy = AverageAccuracy()
        forgetting_measure = ForgettingMeasure(num_tasks)
        
        # Initialize few-shot trainer
        fewshot_trainer = FewShotTargetTrainer(
            model=target_model,
            confidence_estimator=confidence_estimator,
            memory_buffer=memory_buffer,
            prototype_manager=prototype_manager,
            device=device,
            args=args
        )
        
        # Incremental learning
        seen_classes = []
        
        for task_id in range(num_tasks):
            print(f"\n{'='*60}")
            print(f"TASK {task_id + 1}/{num_tasks}")
            print(f"{'='*60}")
            
            start_class = task_id * args.num_classes_per_task
            end_class = min((task_id + 1) * args.num_classes_per_task, total_classes)
            current_classes = list(range(start_class, end_class))
            seen_classes.extend(current_classes)
            
            print(f"Current classes: {current_classes}")
            print(f"Total seen classes: {seen_classes}")
            
            # Get few-shot data loaders
            labeled_loader, unlabeled_loader, test_loader = get_office31_fewshot_loaders(
                root=args.data_root,
                domain=args.target,
                n_shot=args.n_shot,
                labeled_batch_size=args.target_batch_size,
                unlabeled_batch_size=args.unlabeled_batch_size,
                num_workers=args.num_workers,
                selected_classes=current_classes,
                test_classes=seen_classes,
                strong_aug=args.strong_aug
            )
            
            print(f"Labeled samples: {len(labeled_loader.dataset)}")
            print(f"Unlabeled samples: {len(unlabeled_loader.dataset) if unlabeled_loader else 0}")
            print(f"Test samples: {len(test_loader.dataset)}")
            
            # Train on current task
            task_metrics = fewshot_trainer.train_task(
                task_id=task_id,
                labeled_loader=labeled_loader,
                unlabeled_loader=unlabeled_loader,
                test_loader=test_loader,
                epochs=args.target_epochs,
                current_classes=current_classes,
                seen_classes=seen_classes
            )
            
            # Update metrics
            avg_accuracy.update(task_id, task_metrics['accuracy'])
            forgetting_measure.update(task_id, task_metrics['accuracy'])
            
            # Save checkpoint
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f'target_model_task_{task_id + 1}.pth'
            )
            fewshot_trainer.save_checkpoint(checkpoint_path, task_id)
            
            # Print results
            print(f"\nTask {task_id + 1} Results:")
            print(f"  Accuracy: {task_metrics['accuracy']:.2f}%")
            print(f"  Average Accuracy: {avg_accuracy.get_average():.2f}%")
            print(f"  Forgetting: {forgetting_measure.get_forgetting():.4f}")
        
        # Final results
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(f"Final Average Accuracy: {avg_accuracy.get_average():.2f}%")
        print(f"Final Forgetting Measure: {forgetting_measure.get_forgetting():.4f}")

        # Per-task accuracies
        task_accuracies = avg_accuracy.get_all()
        print(f"\nPer-Task Accuracies:")
        for i, acc in enumerate(task_accuracies):
            print(f"  Task {i+1}: {acc:.2f}%")

        # Save metrics
        metrics_path = os.path.join(args.log_dir, 'final_metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write(f"Few-Shot CIDA Experiment: {exp_name}\n")
            f.write(f"Source: {args.source} -> Target: {args.target}\n")
            f.write(f"Few-shot setting: {args.n_shot}-shot\n")
            f.write(f"Final Average Accuracy: {avg_accuracy.get_average():.2f}%\n")
            f.write(f"Final Forgetting: {forgetting_measure.get_forgetting():.4f}\n")
            f.write(f"\nPer-Task Accuracies:\n")
            for i, acc in enumerate(task_accuracies):
                f.write(f"  Task {i+1}: {acc:.2f}%\n")
        
        print(f"\nMetrics saved to: {metrics_path}")


if __name__ == '__main__':
    main()
