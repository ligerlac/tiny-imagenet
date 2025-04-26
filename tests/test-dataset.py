import os
import pytest
import shutil
import tempfile
import torch
from torchvision import transforms
from pathlib import Path
from PIL import Image

try:
    from tiny_imagenet_torch import TinyImageNet
except ImportError:
    # Handle the case where the package isn't installed
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from tiny_imagenet_torch import TinyImageNet

# Use a shared test directory for all tests
TEST_DATA_DIR = Path(tempfile.gettempdir()) / "tiny_imagenet_test_data"

# Global dataset variables
train_dataset = None
test_dataset = None

def setup_module():
    """Download dataset once for all tests."""
    global train_dataset, test_dataset
    
    # Create the test directory if it doesn't exist
    TEST_DATA_DIR.mkdir(exist_ok=True)
    
    # Check if dataset already exists
    if not (TEST_DATA_DIR / "tiny-imagenet-200").exists():
        print(f"\nDownloading TinyImageNet dataset to {TEST_DATA_DIR}...")
        # Download the dataset
        train_dataset = TinyImageNet(
            root=str(TEST_DATA_DIR),
            train=True,
            download=True
        )
    else:
        print(f"\nTinyImageNet dataset already exists at {TEST_DATA_DIR}")
    
    # Create datasets for testing
    train_dataset = TinyImageNet(
        root=str(TEST_DATA_DIR),
        train=True,
        transform=transforms.ToTensor()
    )
    
    test_dataset = TinyImageNet(
        root=str(TEST_DATA_DIR),
        train=False,
        transform=transforms.ToTensor()
    )


class TestTinyImageNet:
    def test_dataset_lengths(self):
        """Test that the dataset has the expected number of samples."""
        # TinyImageNet has 100,000 training images (500 per class × 200 classes)
        # and the test set is not labeled, but has 10,000 images
        assert len(train_dataset) == 100000
        assert len(test_dataset) == 10000
    
    def test_class_count(self):
        """Test that the dataset has the expected number of classes."""
        # TinyImageNet has 200 classes
        assert len(train_dataset.class_to_idx) == 200
    
    def test_image_shape(self):
        """Test that images have the expected shape."""
        # TinyImageNet images are 64x64x3
        img, _ = train_dataset[0]
        assert img.shape == (3, 64, 64)
        assert isinstance(img, torch.Tensor)
        
        # Check a few more images to be sure
        for i in range(1, 5):
            img, _ = train_dataset[i]
            assert img.shape == (3, 64, 64)
    
    def test_targets_type(self):
        """Test that targets have the expected type."""
        # Targets should be integers
        _, target = train_dataset[0]
        assert isinstance(target, int)
        assert 0 <= target < 200  # Class index should be between 0 and 199
    
    def test_transform(self):
        """Test that transforms are applied correctly."""
        # Create a dataset with a resize transform
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])
        
        resized_dataset = TinyImageNet(
            root=str(TEST_DATA_DIR),
            train=True,
            transform=transform
        )
        
        # Check that the transform was applied
        img, _ = resized_dataset[0]
        assert img.shape == (3, 32, 32)
    
    def test_class_to_idx_mapping(self):
        """Test that the class_to_idx mapping is correct."""
        # Check that class_to_idx has the expected format
        assert all(isinstance(k, str) for k in train_dataset.class_to_idx.keys())
        assert all(isinstance(v, int) for v in train_dataset.class_to_idx.values())
        assert all(0 <= v < 200 for v in train_dataset.class_to_idx.values())
    
    def test_target_distribution(self):
        """Test that the class distribution is balanced."""
        # Sample a subset of the dataset to check class distribution
        targets = [train_dataset.targets[i].item() for i in range(0, len(train_dataset), 100)]
        
        # Count the frequency of each class
        class_counts = {}
        for target in targets:
            if target not in class_counts:
                class_counts[target] = 0
            class_counts[target] += 1
        
        # Check that we have a reasonable distribution of classes
        assert len(class_counts) > 100  # Should have many different classes in our sample
    