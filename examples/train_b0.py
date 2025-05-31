import os
import sys
import logging
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.variants import ViscarusB0
from src.training.trainer import ViscarusTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_model_parameters(model):
    """Print detailed parameter statistics for the model."""
    logger.info("\nModel Parameter Statistics:")
    logger.info("-" * 50)
    
    # Get all named parameters
    named_params = OrderedDict(model.named_parameters())
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total Parameters: {total_params:,}")
    logger.info(f"Trainable Parameters: {trainable_params:,}")
    logger.info(f"Non-trainable Parameters: {total_params - trainable_params:,}")
    
    # Print parameters by layer
    logger.info("\nParameters by Layer:")
    logger.info("-" * 50)
    
    current_module = ""
    module_params = 0
    
    for name, param in named_params.items():
        # Get the module name (first part before the first dot)
        module_name = name.split('.')[0]
        
        if module_name != current_module:
            if current_module:
                logger.info(f"{current_module}: {module_params:,} parameters")
            current_module = module_name
            module_params = 0
        
        module_params += param.numel()
    
    # Print the last module
    if current_module:
        logger.info(f"{current_module}: {module_params:,} parameters")
    
    logger.info("-" * 50)

def load_cifar10_data(batch_size=32):
    """Load CIFAR-10 dataset with transformations."""
    logger.info("Loading CIFAR-10 dataset...")
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    try:
        trainset = torchvision.datasets.CIFAR10(
            root='./examples/data', 
            train=True,
            download=True, 
            transform=transform_train
        )
        trainloader = DataLoader(
            trainset, 
            batch_size=batch_size,
            shuffle=True, 
            num_workers=2
        )

        testset = torchvision.datasets.CIFAR10(
            root='./examples/data', 
            train=False,
            download=True, 
            transform=transform_test
        )
        testloader = DataLoader(
            testset, 
            batch_size=batch_size,
            shuffle=False, 
            num_workers=2
        )
        
        logger.info(f"Dataset loaded successfully. Training samples: {len(trainset)}, Test samples: {len(testset)}")
        return trainloader, testloader
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def plot_training_history(history):
    """Plot training and validation metrics."""
    logger.info("Plotting training history...")
    
    try:
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history['train_accuracy'], label='Train Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        
        # Create output directory if it doesn't exist
        os.makedirs('examples/output', exist_ok=True)
        plt.savefig('examples/output/training_history.png')
        plt.close()
        logger.info("Training history plot saved successfully")
        
    except Exception as e:
        logger.error(f"Error plotting training history: {str(e)}")
        raise

def main():
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        # Load data with optimal batch size for 8M parameter model
        trainloader, testloader = load_cifar10_data(batch_size=96)  # Adjusted for larger model

        # Create model with parameters optimized for ~8M parameters
        logger.info("Creating ViscarusB0 model with ~8M parameters...")
        model = ViscarusB0(
            num_classes=10,  # CIFAR-10 has 10 classes
            task_domain='classification'
        )

        # Print detailed parameter statistics
        print_model_parameters(model)

        # Create trainer with stable parameters to prevent NaN
        logger.info("Setting up trainer...")
        trainer = ViscarusTrainer(
            model=model,
            criterion=nn.CrossEntropyLoss(),
            device=device,
            fairness_weight=0.0,  # No fairness constraints for basic training
            optimizer=torch.optim.AdamW(
                model.parameters(),
                lr=5e-4,  # Lower initial learning rate for stability
                weight_decay=0.01,  # Moderate weight decay
                betas=(0.9, 0.95),  # Lower beta2 for better stability
                eps=1e-8
            )
        )

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Add one-cycle learning rate scheduler for better convergence
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            trainer.optimizer,
            max_lr=1e-3,
            epochs=30,
            steps_per_epoch=len(trainloader),
            pct_start=0.3,  # Warm up for 30% of training
            div_factor=25,  # Initial lr = max_lr/25
            final_div_factor=1000  # Final lr = initial_lr/1000
        )

        # Train model with optimized settings for 8M parameter model
        logger.info("Starting training...")
        history = trainer.train(
            train_loader=trainloader,
            val_loader=testloader,
            num_epochs=150,  # More epochs for larger model
            early_stopping_patience=20,  # Increased patience
            scheduler=scheduler  # Add scheduler to trainer
        )

        # Plot training history
        plot_training_history(history)

        # Save model with complete training state
        logger.info("Saving model...")
        os.makedirs('examples/output', exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history,
            'total_params': sum(p.numel() for p in model.parameters())
        }, 'examples/output/viscarus_b0_cifar10_8M.pth')

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        raise

if __name__ == '__main__':
    main() 