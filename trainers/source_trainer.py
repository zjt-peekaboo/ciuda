# trainers/source_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.losses import TripletLoss, WeightedCrossEntropyLoss


class SourceTrainer:
    """Trainer for source domain with Triplet Loss"""
    def __init__(self, model, device, args):
        self.model = model
        self.device = device
        self.args = args
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.triplet_loss = TripletLoss(
            margin=args.triplet_margin,
            mining='hard'
        )
        
        # Optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=args.source_lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=15,
            gamma=0.1
        )
        
    def train(self, train_loader, epochs):
        """Train source model"""
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0.0
            total_ce_loss = 0.0
            total_triplet_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits, features = self.model(images, return_features=True)
                
                # Compute losses
                ce_loss = self.ce_loss(logits, labels)
                triplet_loss = self.triplet_loss(features, labels)
                
                # Combined loss
                loss = ce_loss + \
                       self.args.alpha_triplet * triplet_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                total_ce_loss += ce_loss.item()
                total_triplet_loss += triplet_loss.item()
                
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
            
            # Epoch statistics
            avg_loss = total_loss / len(train_loader)
            avg_ce_loss = total_ce_loss / len(train_loader)
            avg_triplet_loss = total_triplet_loss / len(train_loader)
            accuracy = 100. * correct / total
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Total Loss: {avg_loss:.4f}')
            print(f'  CE Loss: {avg_ce_loss:.4f}')
            print(f'  Triplet Loss: {avg_triplet_loss:.4f}')
            print(f'  Accuracy: {accuracy:.2f}%')
            
            # Update scheduler
            self.scheduler.step()
    
    def save_checkpoint(self, path):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        torch.save(checkpoint, path)





