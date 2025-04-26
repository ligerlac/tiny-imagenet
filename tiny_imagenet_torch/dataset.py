import os
import torch
import numpy as np
from PIL import Image
import zipfile
from tqdm import tqdm
from torchvision.datasets.utils import check_integrity, download_url
from torchvision.datasets.vision import VisionDataset

class TinyImageNet(VisionDataset):
    """TinyImageNet Dataset.
    
    Args:
        root (string): Root directory of dataset.
        train (bool, optional): If True, creates dataset from training and validation set,
            otherwise creates from test set. Default is True.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    filename = "tiny-imagenet-200.zip"
    base_folder = "tiny-imagenet-200"
    md5 = "90528d7ca1a48142e341f4ef8d21d0de"
    
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(TinyImageNet, self).__init__(
            root, transform=transform, target_transform=target_transform
        )
        
        self.train = train
        
        if download:
            self._download()
            
        if not self._check_exists():
            raise RuntimeError('Dataset not found. Use download=True to download')
            
        # Create class to index mapping
        self.class_to_idx = {}
        with open(os.path.join(self.root, self.base_folder, 'wnids.txt'), 'r') as f:
            for i, line in enumerate(f.readlines()):
                self.class_to_idx[line.strip()] = i
        
        # Store paths and targets without loading the images
        self.data, self.targets = self._get_image_paths_and_targets()
        
    def _check_exists(self):
        """Check if the dataset exists and is properly structured"""
        return (os.path.exists(os.path.join(self.root, self.base_folder)) and
                os.path.exists(os.path.join(self.root, self.base_folder, 'wnids.txt')) and
                os.path.exists(os.path.join(self.root, self.base_folder, 'train')) and
                os.path.exists(os.path.join(self.root, self.base_folder, 'val')))
        
    def _download(self):
        """Download and extract the TinyImageNet dataset if it doesn't exist"""
        if self._check_exists():
            print('Files already downloaded and verified')
            return
        
        os.makedirs(self.root, exist_ok=True)
        
        # Download the dataset
        print(f'Downloading {self.url}')
        download_url(self.url, self.root, self.filename, md5=self.md5)
        
        # Extract the dataset
        print(f'Extracting {os.path.join(self.root, self.filename)}')
        with zipfile.ZipFile(os.path.join(self.root, self.filename), 'r') as zip_ref:
            zip_ref.extractall(self.root)
            
        # Clean up
        os.remove(os.path.join(self.root, self.filename))
        print('Download completed successfully')
        
    def _get_image_paths_and_targets(self):
        """Get lists of image paths and corresponding labels"""
        image_paths = []
        targets = []
        
        # Get dataset base path
        base_path = os.path.join(self.root, self.base_folder)
        
        if self.train:
            # Process training data
            for class_id in self.class_to_idx.keys():
                class_index = self.class_to_idx[class_id]
                target_dir = os.path.join(base_path, 'train', class_id, 'images')
                if os.path.isdir(target_dir):
                    for filename in os.listdir(target_dir):
                        if filename.endswith('.JPEG'):
                            path = os.path.join(target_dir, filename)
                            image_paths.append(path)
                            targets.append(class_index)
        else:
            # Process validation data (we ignore test data as it has no labels)
            val_annotations = os.path.join(base_path, 'val', 'val_annotations.txt')
            with open(val_annotations, 'r') as f:
                for line in f.readlines():
                    items = line.strip().split()
                    filename = items[0]
                    class_id = items[1]
                    path = os.path.join(base_path, 'val', 'images', filename)
                    image_paths.append(path)
                    targets.append(self.class_to_idx[class_id])
        
        # Convert targets to tensor for easy filtering
        return image_paths, torch.tensor(targets)

    def __getitem__(self, index):
        """
        Args:
            index: Index of the data point
        Returns:
            tuple: (image, target)
        """
        path, target = self.data[index], self.targets[index].item()
        
        # Load image
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target

    def __len__(self):
        return len(self.data)
