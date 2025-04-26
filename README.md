# tiny-imagenet-torch

A PyTorch-compatible implementation of the TinyImageNet dataset, following the pattern of torchvision datasets like MNIST, FashionMNIST, and CIFAR-10.

## About TinyImageNet

Tiny ImageNet contains 100000 images of 200 classes (500 for each class) downsized to 64×64 colored images. Each class has 500 training images, 50 validation images and 50 test images. More information can be found here: https://paperswithcode.com/dataset/tiny-imagenet. This implementation ingores the unlabeled test images to match the structure of MNIST, FashionMNIST, and CIFAR-10. 

## Installation

```bash
pip install tiny-imagenet-torch
```

Or install from source:

```bash
git clone https://github.com/ligerlac/tiny-imagenet-torch.git
cd tiny-imagenet-torch
pip install -e .
```

## Usage

```python
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tiny_imagenet_torch import TinyImageNet

# Simple transformation - just convert to tensor
transform = transforms.ToTensor()

# Create dataset
train_dataset = TinyImageNet(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = TinyImageNet(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

# Usage example
for images, labels in train_loader:
    # Your training code here
    pass
```

## Dataset Details

- 200 classes from ImageNet
- 500 training images per class (100,000 total)
- 50 validation images per class (10,000 total)
- 50 test images per class (10,000 total)
- All images are 64×64 color images

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The TinyImageNet dataset was created by Stanford for the CS231N course
- The implementation follows torchvision's dataset pattern