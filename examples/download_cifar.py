import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

def download_cifar10(data_dir='data/cifar10'):
    """
    Download and prepare CIFAR-10 dataset
    
    Args:
        data_dir (str): Directory to store the dataset
        
    Returns:
        tuple: (train_loader, test_loader, classes)
    """
    # Create data directory if it doesn't exist
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Define transformations
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

    # Download and load training data
    print("Downloading CIFAR-10 training data...")
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, 
        train=True,
        download=True, 
        transform=transform_train
    )
    
    # Download and load test data
    print("Downloading CIFAR-10 test data...")
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, 
        train=False,
        download=True, 
        transform=transform_test
    )

    # Create data loaders
    train_loader = DataLoader(
        trainset, 
        batch_size=128, 
        shuffle=True, 
        num_workers=2
    )
    
    test_loader = DataLoader(
        testset, 
        batch_size=128, 
        shuffle=False, 
        num_workers=2
    )

    # Get class names
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    print(f"\nDataset downloaded and prepared successfully!")
    print(f"Training samples: {len(trainset)}")
    print(f"Test samples: {len(testset)}")
    print(f"Classes: {classes}")
    
    return train_loader, test_loader, classes

def save_sample_images(data_dir='data/cifar10', num_samples=5):
    """
    Save sample images from each class
    
    Args:
        data_dir (str): Directory to store the dataset
        num_samples (int): Number of samples to save per class
    """
    # Load dataset without transformations
    dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    
    # Create samples directory
    samples_dir = Path(data_dir) / 'samples'
    samples_dir.mkdir(exist_ok=True)
    
    # Save sample images for each class
    for class_idx in range(10):
        class_samples = [i for i, (_, label) in enumerate(dataset) if label == class_idx][:num_samples]
        
        for i, idx in enumerate(class_samples):
            img, _ = dataset[idx]
            img = transforms.ToPILImage()(img)
            img.save(samples_dir / f'class_{class_idx}_sample_{i}.png')
    
    print(f"\nSaved {num_samples} sample images per class in {samples_dir}")

if __name__ == '__main__':
    # Download and prepare dataset
    train_loader, test_loader, classes = download_cifar10()
    
    # Save sample images
    save_sample_images()
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Number of classes: {len(classes)}")
    print(f"Image size: 32x32x3")
    print(f"Training batch size: {train_loader.batch_size}")
    print(f"Test batch size: {test_loader.batch_size}") 