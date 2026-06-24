import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

from src.data.dataset import load_image

class DINODataAugmentation:
    """
    DINO Multi-Crop Data Augmentation.
    Generates:
    - 2 Global views (default 224x224, scale range e.g. 0.4 to 1.0)
    - N Local views (default 96x96, scale range e.g. 0.05 to 0.4)
    """
    def __init__(self, global_crops_scale=(0.4, 1.0), local_crops_scale=(0.05, 0.4), 
                 local_crops_number=6, input_size=224, local_size=96):
        
        # 1. First Global View: with color jitter and Gaussian blur
        self.global_transform_1 = T.Compose([
            T.ToPILImage(),
            T.RandomResizedCrop(input_size, scale=global_crops_scale, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            T.ToTensor(),
        ])
        
        # 2. Second Global View: with color jitter and Gaussian blur
        self.global_transform_2 = T.Compose([
            T.ToPILImage(),
            T.RandomResizedCrop(input_size, scale=global_crops_scale, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            T.ToTensor(),
        ])
        
        # 3. Local Views: color jitter but no Gaussian blur (keeps high-freq features)
        self.local_crops_number = local_crops_number
        self.local_transform = T.Compose([
            T.ToPILImage(),
            T.RandomResizedCrop(local_size, scale=local_crops_scale, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            T.ToTensor(),
        ])

    def __call__(self, image):
        crops = []
        # Add 2 global views
        crops.append(self.global_transform_1(image))
        crops.append(self.global_transform_2(image))
        # Add N local views
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(image))
        return crops

class AntiSpoofingSSLDataset(Dataset):
    """
    Self-Supervised Dataset for Anti-Spoofing.
    Loads images from standard folders (live, spoof) in the train directory and returns crop lists.
    """
    def __init__(self, root_dir, split="train", transform=None):
        self.split_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.samples = []
        
        # Scan live and spoof folders in the split directory
        for class_name in ["live", "spoof"]:
            class_dir = os.path.join(self.split_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for root, _, files in os.walk(class_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        self.samples.append(os.path.join(root, file))
                        
        print(f"SSL Dataset: Loaded {len(self.samples)} images from {self.split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        image = load_image(path)
        if self.transform is not None:
            image = self.transform(image)
        return image
