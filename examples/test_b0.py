import torch
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
import logging
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.variants import ViscarusB0

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_cifar10_test(batch_size=64):
    """Load CIFAR-10 test dataset."""
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

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
    
    return testloader, testset.classes

def visualize_predictions(images, labels, predictions, class_names, num_samples=5):
    """Visualize model predictions."""
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        # Denormalize
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        plt.title(f'True: {class_names[labels[i]]}\nPred: {class_names[predictions[i]]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('examples/output/test_predictions.png')
    plt.close()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load the test data
    testloader, class_names = load_cifar10_test()
    logger.info("Test data loaded successfully")

    # Load the model
    try:
        logger.info("Loading saved model...")
        model = ViscarusB0(num_classes=10)
        checkpoint = torch.load('examples/output/viscarus_b0.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return

    # Test metrics
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    test_images = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)['predictions']
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Store some images for visualization
            if batch_idx == 0:
                test_images = images[:5].cpu()
                all_labels = labels[:5].cpu()
                all_predictions = predicted[:5].cpu()

    accuracy = 100. * correct / total
    logger.info(f'Test Accuracy: {accuracy:.2f}%')

    # Visualize some predictions
    visualize_predictions(test_images, all_labels, all_predictions, class_names)
    logger.info("Predictions visualization saved to examples/output/test_predictions.png")

if __name__ == '__main__':
    main()
