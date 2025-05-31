import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import time
import json
from datetime import datetime

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

def train_all_models():
    # Download CIFAR-10 dataset
    train_loader, test_loader, classes = download_cifar10()
    
    # Model configurations
    model_configs = [
        # B1 model (non-multimodal)
        {
            'name': 'ViscarusB1',
            'class': ViscarusB1,
            'is_multimodal': False,
            'epochs': 100,
            'lr': 0.001,
            'weight_decay': 0.01
        },
        # B2-B7 models (multimodal)
        {
            'name': 'ViscarusMultiModalB2',
            'class': ViscarusMultiModalB2,
            'is_multimodal': True,
            'epochs': 120,
            'lr': 0.0008,
            'weight_decay': 0.008
        },
        {
            'name': 'ViscarusMultiModalB3',
            'class': ViscarusMultiModalB3,
            'is_multimodal': True,
            'epochs': 140,
            'lr': 0.0006,
            'weight_decay': 0.006
        },
        {
            'name': 'ViscarusMultiModalB4',
            'class': ViscarusMultiModalB4,
            'is_multimodal': True,
            'epochs': 160,
            'lr': 0.0005,
            'weight_decay': 0.005
        },
        {
            'name': 'ViscarusMultiModalB5',
            'class': ViscarusMultiModalB5,
            'is_multimodal': True,
            'epochs': 180,
            'lr': 0.0004,
            'weight_decay': 0.004
        },
        {
            'name': 'ViscarusMultiModalB6',
            'class': ViscarusMultiModalB6,
            'is_multimodal': True,
            'epochs': 200,
            'lr': 0.0003,
            'weight_decay': 0.003
        },
        {
            'name': 'ViscarusMultiModalB7',
            'class': ViscarusMultiModalB7,
            'is_multimodal': True,
            'epochs': 220,
            'lr': 0.0002,
            'weight_decay': 0.002
        }
    ]
    
    # Results storage
    results = []
    
    # Train each model
    for config in model_configs:
        print(f"\n{'='*50}")
        print(f"Starting training for {config['name']}")
        print(f"{'='*50}")
        
        # Initialize model
        model = config['class']()
        
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
            'best_accuracy': best_acc,
            'parameters': sum(p.numel() for p in model.parameters()) / 1e6,
            'epochs': config['epochs']
        })
        
        # Save results after each model
        with open('output/training_results.json', 'w') as f:
            json.dump(results, f, indent=4)
    
    # Print final summary
    print("\nTraining Summary:")
    print("=" * 50)
    for result in results:
        print(f"\nModel: {result['model']}")
        print(f"Best Accuracy: {result['best_accuracy']:.2f}%")
        print(f"Parameters: {result['parameters']:.2f}M")
        print(f"Training Epochs: {result['epochs']}")
        print("-" * 30)

if __name__ == '__main__':
    train_all_models() 