from typing import Dict, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import logging
from tqdm import tqdm
import time
import os
import sys
from pathlib import Path
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import math
import numpy as np

def is_valid_loss(loss_tensor):
    """Check if loss value is valid (not NaN or inf)."""
    return not torch.isnan(loss_tensor).any() and not torch.isinf(loss_tensor).any()

def get_grad_norm(parameters):
    """Calculate gradient norm for monitoring."""
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

# Add the src directory to the Python path
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from models.variants import ViscarusB1

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ViscarusB1Trainer:
    """
    Trainer for ViscarusB1 model with 15M parameters.
    Optimized for efficient training with balanced memory usage.
    """
    def __init__(
        self,
        model: ViscarusB1,
        criterion: nn.Module = nn.CrossEntropyLoss(),
        optimizer: optim.Optimizer = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 2e-4,  # Adjusted for 15M params
        weight_decay: float = 1e-5    # Adjusted for 15M params
    ):
        self.model = model
        self.criterion = criterion
        self.device = device
        
        # Set up optimizer if not provided
        if optimizer is None:
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        else:
            self.optimizer = optimizer
        
        # Move model to device
        self.model.to(device)
        
        # Training metrics
        self.best_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        total_epochs: int
    ) -> Dict[str, float]:
        """
        Train for one epoch with gradient accumulation for memory efficiency.
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Create progress bar
        pbar = tqdm(
            train_loader,
            desc=f'Epoch {epoch + 1}/{total_epochs} [Train]',
            leave=True,
            ncols=100
        )
        
        start_time = time.time()
        
        # Gradient accumulation steps
        accumulation_steps = 2  # Accumulate gradients for 2 steps
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            predictions = outputs['predictions']
            
            # Calculate loss
            loss = self.criterion(predictions, targets)
            loss = loss / accumulation_steps  # Normalize loss
            
            # Backward pass with gradient accumulation
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                # Update weights
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item() * accumulation_steps
            _, predicted = predictions.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item() * accumulation_steps:.4f}',
                'acc': f'{100.*correct/total:.2f}%',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        # Calculate epoch metrics
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        epoch_time = time.time() - start_time
        
        # Store metrics
        self.train_losses.append(epoch_loss)
        self.train_accs.append(epoch_acc)
        
        # Print epoch summary
        logger.info(f'\nEpoch {epoch + 1}/{total_epochs} Summary:')
        logger.info(f'Time: {epoch_time:.2f}s | '
                   f'Loss: {epoch_loss:.4f} | '
                   f'Acc: {epoch_acc:.2f}% | '
                   f'LR: {self.optimizer.param_groups[0]["lr"]:.2e}')
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'time': epoch_time
        }
    
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int,
        total_epochs: int
    ) -> Dict[str, float]:
        """
        Validate the model with mixed precision for efficiency.
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        # Create progress bar
        pbar = tqdm(
            val_loader,
            desc=f'Epoch {epoch + 1}/{total_epochs} [Val]',
            leave=True,
            ncols=100
        )
        
        start_time = time.time()
        
        with torch.no_grad():
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                predictions = outputs['predictions']
                
                # Calculate loss
                loss = self.criterion(predictions, targets)
                
                # Update metrics
                total_loss += loss.item()
                _, predicted = predictions.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        # Calculate validation metrics
        val_loss = total_loss / len(val_loader)
        val_acc = 100. * correct / total
        val_time = time.time() - start_time
        
        # Store metrics
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)
        
        # Print validation summary
        logger.info(f'Validation Summary:')
        logger.info(f'Time: {val_time:.2f}s | '
                   f'Loss: {val_loss:.4f} | '
                   f'Acc: {val_acc:.2f}%')
        
        return {
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_time': val_time
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_dir: str = 'checkpoints',
        early_stopping_patience: int = 5,
        scheduler: optim.lr_scheduler._LRScheduler = None
    ) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs with advanced optimization.
        """
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train for one epoch
            train_metrics = self.train_epoch(train_loader, epoch, num_epochs)
            
            # Validate
            val_metrics = self.validate(val_loader, epoch, num_epochs)
            
            # Update learning rate
            if scheduler is not None:
                scheduler.step(val_metrics['val_accuracy'])
            
            # Save best model
            if val_metrics['val_accuracy'] > best_val_acc:
                best_val_acc = val_metrics['val_accuracy']
                patience_counter = 0
                
                # Save checkpoint
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'train_accs': self.train_accs,
                    'val_accs': self.val_accs
                }
                torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
                logger.info(f'Saved best model with validation accuracy: {best_val_acc:.2f}%')
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f'Early stopping triggered after {epoch + 1} epochs')
                break
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'best_val_acc': best_val_acc
        }

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize wandb
    wandb.init(project="viscarus-cifar10", name="efficientnet-b1-cifar10")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Data transforms - minimal augmentation for stability
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform_train
    )
    
    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform_test
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=256,  # Increased batch size to better utilize 15M params
        shuffle=True,
        num_workers=4,  # Increased workers for faster data loading
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,  # Increased batch size
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = ViscarusB1(
        num_classes=10,  # CIFAR-10 has 10 classes
        task_domain=None  # No specific task domain for CIFAR-10
    )
    model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Increased label smoothing for better generalization
    
    # Layer-wise learning rate decay for better parameter utilization
    param_groups = []
    base_lr = 0.001  # Higher base learning rate for 15M params
    
    # Group parameters by layer depth
    for name, param in model.named_parameters():
        if 'bias' in name:
            param_groups.append({'params': [param], 'lr': base_lr, 'weight_decay': 0})
        else:
            # Apply smaller learning rates to deeper layers
            depth = name.count('.')
            layer_lr = base_lr * (0.9 ** depth)  # Decay rate of 0.9 per layer depth
            param_groups.append({'params': [param], 'lr': layer_lr, 'weight_decay': 0.01})
    
    optimizer = optim.AdamW(
        param_groups,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler with warmup
    num_warmup_steps = 5 * len(train_loader)  # 5 epochs of warmup for 15M params
    num_training_steps = 100 * len(train_loader)  # 100 epochs total
    
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(num_training_steps - num_warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    num_epochs = 100
    best_acc = 0.0
    scaler = torch.amp.GradScaler()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision training
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(inputs)['predictions']
                loss = criterion(outputs, targets)
            
            # Check for NaN loss
            if torch.isnan(loss):
                logger.warning(f'NaN loss detected in training at batch {batch_idx}')
                continue
            
            # Scale loss and backpropagate
            scaler.scale(loss).backward()
            
            # Unscale gradients for gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Increased for 15M params
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            
            # Update learning rate after optimizer step
            scheduler.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': train_loss/(batch_idx+1),
                'acc': 100.*train_correct/train_total,
                'lr': optimizer.param_groups[0]['lr']
            })
        
        # Validation phase
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        valid_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Test]')
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Use mixed precision for validation too
                with torch.amp.autocast(device_type='cuda'):  # Fixed autocast API
                    outputs = model(inputs)['predictions']
                    loss = criterion(outputs, targets)
                
                # Check for NaN values
                if torch.isnan(loss):
                    logger.warning(f'NaN loss detected in validation at batch {batch_idx}')
                    continue
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
                valid_batches += 1
                
                pbar.set_postfix({
                    'loss': test_loss/valid_batches if valid_batches > 0 else 0,
                    'acc': 100.*test_correct/test_total if test_total > 0 else 0
                })
        
        # Calculate epoch metrics
        train_acc = 100.*train_correct/train_total if train_total > 0 else 0
        test_acc = 100.*test_correct/test_total if test_total > 0 else 0
        avg_test_loss = test_loss/valid_batches if valid_batches > 0 else float('inf')
        
        # Log metrics
        wandb.log({
            'train_loss': train_loss/len(train_loader),
            'train_acc': train_acc,
            'test_loss': avg_test_loss,
            'test_acc': test_acc,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
            }, 'best_model.pth')
            logger.info(f'New best accuracy: {best_acc:.2f}%')
        
        logger.info(f'Epoch {epoch+1}/{num_epochs}: '
                   f'Train Loss: {train_loss/len(train_loader):.4f}, '
                   f'Train Acc: {train_acc:.2f}%, '
                   f'Test Loss: {avg_test_loss:.4f}, '
                   f'Test Acc: {test_acc:.2f}%')
    
    wandb.finish()

if __name__ == '__main__':
    main()