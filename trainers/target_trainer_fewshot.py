# trainers/target_trainer_fewshot.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from utils.losses import TripletLoss, WeightedCrossEntropyLoss, ContrastiveLoss

class FewShotTargetTrainer:
    """
    Trainer for few-shot target domain adaptation
    Combines supervised learning on labeled data with semi-supervised learning on unlabeled data
    """
    def __init__(self, model, confidence_estimator, memory_buffer, 
                 prototype_manager, device, args):
        self.model = model
        self.confidence_estimator = confidence_estimator
        self.memory_buffer = memory_buffer
        self.prototype_manager = prototype_manager
        self.device = device
        self.args = args
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.weighted_ce_loss = WeightedCrossEntropyLoss()
        self.triplet_loss = TripletLoss(margin=args.triplet_margin, mining='semi-hard')
        self.contrastive_loss = ContrastiveLoss(temperature=0.07)
        
    def train_task(self, task_id, labeled_loader, unlabeled_loader, test_loader,
                   epochs, current_classes, seen_classes):
        """Train on a single few-shot task"""
        
        # Initialize optimizer with lower learning rate for fine-tuning
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.args.target_lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=self.args.target_lr * 0.01
        )
        
        # Initialize prototypes with labeled data
        self._initialize_prototypes(labeled_loader, current_classes)
        
        best_acc = 0.0
        best_epoch = 0
        
        for epoch in range(epochs):
            # Training phase
            train_metrics = self._train_epoch(
                epoch, labeled_loader, unlabeled_loader, optimizer,
                current_classes, seen_classes
            )
            
            # Evaluation phase
            test_acc = self.evaluate(test_loader, seen_classes)
            
            # Track best model
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch
                self._save_best_model(task_id)
            
            # Update learning rate
            scheduler.step()
            
            # Print progress
            print(f'Epoch [{epoch+1}/{epochs}] '
                  f'Train Acc: {train_metrics["train_acc"]:.2f}% '
                  f'Test Acc: {test_acc:.2f}% '
                  f'Best: {best_acc:.2f}% (Epoch {best_epoch+1})')
        
        # Load best model
        self._load_best_model(task_id)
        final_acc = self.evaluate(test_loader, seen_classes)
        
        # Update memory buffer
        self._update_memory_buffer(labeled_loader, unlabeled_loader, current_classes)
        
        return {'accuracy': final_acc}
    
    def _initialize_prototypes(self, labeled_loader, current_classes):
        """Initialize prototypes from labeled samples"""
        self.model.eval()
        
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in labeled_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                features = self.model.get_features(images)
                all_features.append(features)
                all_labels.append(labels)
        
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Initialize prototypes
        self.prototype_manager.initialize_prototypes(all_features, all_labels)
        
        print(f"Initialized prototypes for classes: {current_classes}")
    
    def _train_epoch(self, epoch, labeled_loader, unlabeled_loader, optimizer,
                    current_classes, seen_classes):
        """Train for one epoch with labeled and unlabeled data"""
        self.model.train()
        
        total_loss = 0.0
        total_supervised = 0.0
        total_prototype = 0.0
        total_pseudo = 0.0
        total_consistency = 0.0
        correct = 0
        total = 0
        
        # Create iterator for unlabeled data
        if unlabeled_loader is not None:
            unlabeled_iter = iter(unlabeled_loader)
        else:
            unlabeled_iter = None
        
        pbar = tqdm(labeled_loader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, (labeled_images, labeled_labels) in enumerate(pbar):
            labeled_images = labeled_images.to(self.device)
            labeled_labels = labeled_labels.to(self.device)
            
            # ==================== SUPERVISED LEARNING ====================
            logits, features = self.model(labeled_images, return_features=True)
            
            # Supervised cross-entropy loss
            supervised_loss = self.ce_loss(logits, labeled_labels)
            
            # Prototype alignment loss
            prototype_loss = self.prototype_manager.get_prototype_loss(
                features, labeled_labels, seen_classes
            )
            
            # Combine losses
            loss = (self.args.beta_supervised * supervised_loss +
                   self.args.beta_prototype * prototype_loss)
            
            # Statistics
            _, predicted = logits.max(1)
            total += labeled_labels.size(0)
            correct += predicted.eq(labeled_labels).sum().item()
            
            # ==================== SEMI-SUPERVISED LEARNING ====================
            pseudo_loss = 0.0
            consistency_loss = 0.0
            
            if unlabeled_iter is not None and epoch >= self.args.warmup_epochs:
                try:
                    if self.args.strong_aug:
                        (weak_images, strong_images), _ = next(unlabeled_iter)
                        weak_images = weak_images.to(self.device)
                        strong_images = strong_images.to(self.device)
                    else:
                        unlabeled_images, _ = next(unlabeled_iter)
                        weak_images = unlabeled_images.to(self.device)
                        strong_images = weak_images
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    if self.args.strong_aug:
                        (weak_images, strong_images), _ = next(unlabeled_iter)
                        weak_images = weak_images.to(self.device)
                        strong_images = strong_images.to(self.device)
                    else:
                        unlabeled_images, _ = next(unlabeled_iter)
                        weak_images = unlabeled_images.to(self.device)
                        strong_images = weak_images
                
                # Generate pseudo-labels from weak augmentation
                with torch.no_grad():
                    weak_logits, weak_features = self.model(
                        weak_images, return_features=True
                    )
                    
                    # Combine model prediction and prototype prediction
                    model_confidence, model_pred = self._get_confident_predictions(
                        weak_logits, seen_classes
                    )
                    
                    proto_confidence, proto_pred = self.prototype_manager.compute_prototype_confidence(
                        weak_features, seen_classes, temperature=0.1
                    )
                    
                    # Use ensemble confidence
                    ensemble_confidence = (model_confidence + proto_confidence) / 2
                    
                    # Select high-confidence samples
                    confident_mask = ensemble_confidence > self.args.confidence_threshold
                    
                    # Prefer prototype predictions for borderline cases
                    proto_mask = proto_confidence > self.args.proto_confidence_threshold
                    pseudo_labels = torch.where(proto_mask, proto_pred, model_pred)
                
                if confident_mask.sum() > 0:
                    # Pseudo-label loss on strong augmentation
                    strong_logits = self.model(strong_images[confident_mask])
                    pseudo_loss = self.weighted_ce_loss(
                        strong_logits,
                        pseudo_labels[confident_mask],
                        weights=ensemble_confidence[confident_mask]
                    )
                    
                    loss += self.args.beta_pseudo * pseudo_loss
                
                # Consistency regularization
                if self.args.strong_aug:
                    strong_logits_all = self.model(strong_images)
                    consistency_loss = F.kl_div(
                        F.log_softmax(strong_logits_all[:, seen_classes], dim=1),
                        F.softmax(weak_logits[:, seen_classes].detach(), dim=1),
                        reduction='batchmean'
                    )
                    
                    loss += self.args.beta_consistency * consistency_loss
            
            # ==================== EXPERIENCE REPLAY ====================
            if self.memory_buffer.get_size() > 0:
                replay_loss = self._compute_replay_loss(seen_classes)
                loss += self.args.beta_replay * replay_loss
            
            # ==================== BACKWARD PASS ====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update prototypes with labeled data
            if self.args.update_prototypes and batch_idx % 5 == 0:
                with torch.no_grad():
                    self.prototype_manager.update_prototypes(
                        features.detach(), labeled_labels, use_momentum=True
                    )
            
            # Statistics
            total_loss += loss.item()
            total_supervised += supervised_loss.item()
            total_prototype += prototype_loss.item()
            if isinstance(pseudo_loss, torch.Tensor):
                total_pseudo += pseudo_loss.item()
            if isinstance(consistency_loss, torch.Tensor):
                total_consistency += consistency_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'acc': f'{100.*correct/total:.1f}%',
                'sup': f'{supervised_loss.item():.3f}',
                'proto': f'{prototype_loss.item():.3f}'
            })
        
        metrics = {
            'train_acc': 100. * correct / total,
            'total_loss': total_loss / len(labeled_loader),
            'supervised_loss': total_supervised / len(labeled_loader),
            'prototype_loss': total_prototype / len(labeled_loader),
            'pseudo_loss': total_pseudo / len(labeled_loader),
            'consistency_loss': total_consistency / len(labeled_loader)
        }
        
        return metrics
    
    def _get_confident_predictions(self, logits, seen_classes):
        """Get confident predictions from model"""
        probs = F.softmax(logits[:, seen_classes], dim=1)
        confidence, pred_idx = probs.max(dim=1)
        predictions = torch.tensor(seen_classes, device=logits.device)[pred_idx]
        return confidence, predictions
    
    def _compute_replay_loss(self, seen_classes):
        """Compute loss on memory buffer samples"""
        replay_images, replay_labels = self.memory_buffer.get_all_data()
        
        if replay_images is None:
            return 0.0
        
        # Sample a batch from replay buffer
        if len(replay_images) > self.args.target_batch_size:
            indices = torch.randperm(len(replay_images))[:self.args.target_batch_size]
            replay_images = replay_images[indices]
            replay_labels = replay_labels[indices]
        
        replay_images = replay_images.to(self.device)
        replay_labels = replay_labels.to(self.device)
        
        # Forward pass
        replay_logits, replay_features = self.model(replay_images, return_features=True)
        
        # Cross-entropy loss
        replay_loss = self.ce_loss(replay_logits, replay_labels)
        
        # Prototype alignment for old classes
        proto_loss = self.prototype_manager.get_prototype_loss(
            replay_features, replay_labels, seen_classes
        )
        
        return replay_loss + 0.5 * proto_loss
    
    def _update_memory_buffer(self, labeled_loader, unlabeled_loader, current_classes):
        """Update memory buffer with new task data"""
        self.model.eval()
        
        all_images = []
        all_features = []
        all_labels = []
        all_confidences = []
        all_is_labeled = []
        
        # Collect labeled data (priority)
        with torch.no_grad():
            for images, labels in labeled_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                features = self.model.get_features(images)
                
                all_images.append(images.cpu())
                all_features.append(features.cpu())
                all_labels.append(labels.cpu())
                all_confidences.append(torch.ones(len(labels)))  # Max confidence for labeled
                all_is_labeled.append(torch.ones(len(labels), dtype=torch.bool))
        
        # Collect high-confidence unlabeled data
        if unlabeled_loader is not None:
            with torch.no_grad():
                for data in unlabeled_loader:
                    if self.args.strong_aug:
                        (images, _), _ = data
                    else:
                        images, _ = data
                    
                    images = images.to(self.device)
                    
                    logits, features = self.model(images, return_features=True)
                    
                    # Get pseudo labels with confidence
                    confidence, pseudo_labels = self.prototype_manager.compute_prototype_confidence(
                        features, current_classes
                    )
                    
                    # Only keep high-confidence samples
                    mask = confidence > 0.9
                    
                    if mask.sum() > 0:
                        all_images.append(images[mask].cpu())
                        all_features.append(features[mask].cpu())
                        all_labels.append(pseudo_labels[mask].cpu())
                        all_confidences.append(confidence[mask].cpu())
                        all_is_labeled.append(torch.zeros(mask.sum(), dtype=torch.bool))
        
        if len(all_images) > 0:
            all_images = torch.cat(all_images, dim=0)
            all_features = torch.cat(all_features, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            all_confidences = torch.cat(all_confidences, dim=0)
            all_is_labeled = torch.cat(all_is_labeled, dim=0)
            
            # Update buffer for each class
            for class_id in current_classes:
                class_mask = all_labels == class_id
                
                if class_mask.sum() == 0:
                    continue
                
                class_images = all_images[class_mask]
                class_features = all_features[class_mask]
                class_confidences = all_confidences[class_mask]
                class_is_labeled = all_is_labeled[class_mask]
                
                # Boost confidence for labeled samples
                if self.args.prioritize_labeled:
                    class_confidences = torch.where(
                        class_is_labeled,
                        torch.ones_like(class_confidences) * 2.0,  # Double weight
                        class_confidences
                    )
                
                self.memory_buffer.add_class(
                    class_id, class_images, class_features, class_confidences
                )
        
        print(f"Memory buffer size: {self.memory_buffer.get_size()}")
    
    @torch.no_grad()
    def evaluate(self, test_loader, seen_classes):
        """Evaluate model on test set"""
        self.model.eval()
        
        correct = 0
        total = 0
        
        for images, labels in test_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            logits = self.model(images)
            logits = logits[:, seen_classes]
            
            # Adjust labels
            label_map = {old: new for new, old in enumerate(seen_classes)}
            adjusted_labels = torch.tensor(
                [label_map[l.item()] for l in labels],
                device=labels.device
            )
            
            # Predictions
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(adjusted_labels).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy
    
    def _save_best_model(self, task_id):
        """Save best model during training"""
        self.best_model_state = {
            'model': self.model.state_dict(),
            'prototypes': self.prototype_manager.prototypes.clone()
        }
    
    def _load_best_model(self, task_id):
        """Load best model"""
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state['model'])
            self.prototype_manager.prototypes = self.best_model_state['prototypes']
    
    def save_checkpoint(self, path, task_id):
        """Save checkpoint"""
        checkpoint = {
            'task_id': task_id,
            'model_state_dict': self.model.state_dict(),
            'memory_buffer': self.memory_buffer.buffer,
            'prototypes': self.prototype_manager.prototypes.cpu(),
            'prototype_counts': self.prototype_manager.prototype_counts.cpu(),
            'initialized': self.prototype_manager.initialized.cpu()
        }
        torch.save(checkpoint, path)