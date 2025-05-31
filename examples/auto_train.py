import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import time
import json
from datetime import datetime
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet

from src.models.variants import ViscarusB1, ViscarusB2, ViscarusB3, ViscarusB4, ViscarusB5, ViscarusB6, ViscarusB7
from src.models.multimodal import ViscarusMultiModalB2, ViscarusMultiModalB3, ViscarusMultiModalB4, ViscarusMultiModalB5, ViscarusMultiModalB6, ViscarusMultiModalB7
from examples.download_cifar import download_cifar10

class ModelTrainer:
    def __init__(self, model, model_name, is_multimodal=False):
        self.model = model
        self.model_name = model_name
        self.is_multimodal = is_multimodal
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Create output directory
        self.output_dir = Path(f"output/{model_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epochs': []
        }
    
    def train_epoch(self, train_loader, criterion, optimizer):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def validate(self, test_loader, criterion):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss = running_loss / len(test_loader)
        val_acc = 100. * correct / total
        return val_loss, val_acc
    
    def train(self, train_loader, test_loader, epochs, lr=0.001, weight_decay=0.01):
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_acc = 0
        start_time = time.time()
        
        print(f"\nTraining {self.model_name}...")
        print(f"Device: {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            val_loss, val_acc = self.validate(test_loader, criterion)
            scheduler.step()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['epochs'].append(epoch)
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, self.output_dir / 'best_model.pth')
            
            # Print progress
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Best Val Acc: {best_acc:.2f}%')
            print(f'LR: {scheduler.get_last_lr()[0]:.6f}')
            print('-' * 50)
        
        # Save training history
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f)
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time/3600:.2f} hours")
        print(f"Best validation accuracy: {best_acc:.2f}%")
        
        return best_acc

def get_imagenet_loaders(data_dir='data/imagenet', batch_size=256):
    """Get ImageNet data loaders with appropriate transforms"""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        normalize,
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    train_dataset = ImageNet(
        root=data_dir,
        split='train',
        transform=train_transform
    )
    
    val_dataset = ImageNet(
        root=data_dir,
        split='val',
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader

def train_all_models():
    # Model configurations with their specific parameters
    model_configs = [
        # B1 model (CIFAR-10)
        {
            'name': 'ViscarusB1',
            'class': ViscarusB1,
            'is_multimodal': False,
            'dataset': 'cifar10',
            'epochs': 100,
            'lr': 0.001,
            'weight_decay': 0.01,
            'batch_size': 128,
            'features': {
                'use_attention': True,
                'use_feature_refinement': True,
                'use_cross_layer': True
            }
        },
        # B2 model (CIFAR-10)
        {
            'name': 'ViscarusMultiModalB2',
            'class': ViscarusMultiModalB2,
            'is_multimodal': True,
            'dataset': 'cifar10',
            'epochs': 120,
            'lr': 0.0008,
            'weight_decay': 0.008,
            'batch_size': 128,
            'features': {
                'use_attention': True,
                'use_feature_refinement': True,
                'use_cross_layer': True,
                'use_multimodal': True
            }
        },
        # B3-B7 models (ImageNet)
        {
            'name': 'ViscarusMultiModalB3',
            'class': ViscarusMultiModalB3,
            'is_multimodal': True,
            'dataset': 'imagenet',
            'epochs': 140,
            'lr': 0.0006,
            'weight_decay': 0.006,
            'batch_size': 256,
            'features': {
                'use_attention': True,
                'use_feature_refinement': True,
                'use_cross_layer': True,
                'use_multimodal': True,
                'use_advanced_fusion': True
            }
        },
        {
            'name': 'ViscarusMultiModalB4',
            'class': ViscarusMultiModalB4,
            'is_multimodal': True,
            'dataset': 'imagenet',
            'epochs': 160,
            'lr': 0.0005,
            'weight_decay': 0.005,
            'batch_size': 256,
            'features': {
                'use_attention': True,
                'use_feature_refinement': True,
                'use_cross_layer': True,
                'use_multimodal': True,
                'use_advanced_fusion': True,
                'use_adaptive_pooling': True
            }
        },
        {
            'name': 'ViscarusMultiModalB5',
            'class': ViscarusMultiModalB5,
            'is_multimodal': True,
            'dataset': 'imagenet',
            'epochs': 180,
            'lr': 0.0004,
            'weight_decay': 0.004,
            'batch_size': 256,
            'features': {
                'use_attention': True,
                'use_feature_refinement': True,
                'use_cross_layer': True,
                'use_multimodal': True,
                'use_advanced_fusion': True,
                'use_adaptive_pooling': True,
                'use_dynamic_routing': True
            }
        },
        {
            'name': 'ViscarusMultiModalB6',
            'class': ViscarusMultiModalB6,
            'is_multimodal': True,
            'dataset': 'imagenet',
            'epochs': 200,
            'lr': 0.0003,
            'weight_decay': 0.003,
            'batch_size': 256,
            'features': {
                'use_attention': True,
                'use_feature_refinement': True,
                'use_cross_layer': True,
                'use_multimodal': True,
                'use_advanced_fusion': True,
                'use_adaptive_pooling': True,
                'use_dynamic_routing': True,
                'use_hierarchical_attention': True
            }
        },
        {
            'name': 'ViscarusMultiModalB7',
            'class': ViscarusMultiModalB7,
            'is_multimodal': True,
            'dataset': 'imagenet',
            'epochs': 220,
            'lr': 0.0002,
            'weight_decay': 0.002,
            'batch_size': 256,
            'features': {
                'use_attention': True,
                'use_feature_refinement': True,
                'use_cross_layer': True,
                'use_multimodal': True,
                'use_advanced_fusion': True,
                'use_adaptive_pooling': True,
                'use_dynamic_routing': True,
                'use_hierarchical_attention': True,
                'use_advanced_regularization': True
            }
        }
    ]
    
    # Results storage
    results = []
    
    # Train each model
    for config in model_configs:
        print(f"\n{'='*50}")
        print(f"Starting training for {config['name']}")
        print(f"Dataset: {config['dataset']}")
        print(f"Features: {config['features']}")
        print(f"{'='*50}")
        
        # Get appropriate data loaders
        if config['dataset'] == 'cifar10':
            train_loader, test_loader, _ = download_cifar10()
        else:  # imagenet
            train_loader, test_loader = get_imagenet_loaders(batch_size=config['batch_size'])
        
        # Initialize model with specific features
        model = config['class'](**config['features'])
        
        # Create trainer
        trainer = ModelTrainer(model, config['name'], config['is_multimodal'])
        
        # Train model
        best_acc = trainer.train(
            train_loader,
            test_loader,
            epochs=config['epochs'],
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        
        # Store results
        results.append({
            'model': config['name'],
            'dataset': config['dataset'],
            'best_accuracy': best_acc,
            'parameters': sum(p.numel() for p in model.parameters()) / 1e6,
            'epochs': config['epochs'],
            'features': config['features']
        })
        
        # Save results after each model
        with open('output/training_results.json', 'w') as f:
            json.dump(results, f, indent=4)
    
    # Print final summary
    print("\nTraining Summary:")
    print("=" * 50)
    for result in results:
        print(f"\nModel: {result['model']}")
        print(f"Dataset: {result['dataset']}")
        print(f"Best Accuracy: {result['best_accuracy']:.2f}%")
        print(f"Parameters: {result['parameters']:.2f}M")
        print(f"Training Epochs: {result['epochs']}")
        print("Features:")
        for feature, enabled in result['features'].items():
            print(f"  - {feature}: {enabled}")
        print("-" * 30)

if __name__ == '__main__':
    train_all_models() 