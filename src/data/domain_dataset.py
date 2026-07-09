import os
import cv2
import csv
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# Import utilities, transforms and classes from the original dataset file
from src.data.dataset import (
    load_image,
    generate_FT,
    RandomRotationWithReflect,
    SafeRandAugment,
    AntiSpoofingDataset
)

class TargetDomainDataset(Dataset):
    """
    Dataset for target domain video data.
    Loads image paths and labels from a CSV file if provided, 
    otherwise falls back to recursively walking the directory (treating as unlabeled).
    """
    def __init__(self, root_dir, split="train", csv_path=None, transform=None, return_ft=False, fourier_size=(16, 16)):
        self.split_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.return_ft = return_ft
        self.fourier_size = fourier_size
        
        self.samples = []
        
        if csv_path is not None and os.path.exists(csv_path):
            print(f"TargetDomainDataset: Loading samples from CSV file: {csv_path}")
            with open(csv_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img_rel_path = row.get("path") or row.get("image_path")
                    # Label is required, default to 0 (live) if not present or conversion fails
                    try:
                        label = int(row.get("label", 0))
                    except (ValueError, TypeError):
                        label = 0
                    
                    if img_rel_path:
                        # Handle absolute vs relative paths
                        if os.path.isabs(img_rel_path):
                            img_path = img_rel_path
                        else:
                            # Try joining with root_dir or split directory
                            img_path = os.path.join(root_dir, img_rel_path)
                            if not os.path.exists(img_path):
                                img_path = os.path.join(self.split_dir, img_rel_path)
                                
                        self.samples.append((img_path, label))
        else:
            # Fallback to recursively scanning directories if no CSV is found
            # (Labels are defaulted to -1 to indicate unlabeled target data)
            print(f"TargetDomainDataset: CSV file not found or not provided. Scanning recursively as unlabeled.")
            if os.path.exists(self.split_dir):
                for root, _, files in os.walk(self.split_dir):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                            img_path = os.path.join(root, file)
                            self.samples.append((img_path, -1)) # -1 represents unlabeled/dummy
            else:
                print(f"Warning: Target directory {self.split_dir} does not exist.")
                        
        print(f"TargetDomainDataset: Loaded {len(self.samples)} images for split '{split}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = load_image(path)
        
        if self.return_ft:
            ft_sample = generate_FT(image)
            ft_sample = cv2.resize(ft_sample, self.fourier_size)
            ft_sample = torch.from_numpy(ft_sample).float()
            ft_sample = torch.unsqueeze(ft_sample, 0) # Shape: (1, H, W)
            
        if self.transform is not None:
            image = self.transform(image)
            
        if self.return_ft:
            return image, ft_sample, label
        return image, label


class DomainBalancedDataset(Dataset):
    """
    Wrapper Dataset that combines Source and Target datasets.
    Resolves data imbalance (e.g. 90k source vs 6k target) by wrapping target indices
    using modulo mapping, ensuring a 1:1 balance in each batch.
    
    Returns a dictionary of tensors to prevent unpacking errors under different settings:
    {
        "source_img": Tensor,
        "source_label": Tensor/int,
        "source_ft": Tensor (optional),
        "target_img": Tensor,
        "target_label": Tensor/int,
        "target_ft": Tensor (optional)
    }
    """
    def __init__(self, source_dataset, target_dataset):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        
        if len(self.source_dataset) == 0:
            print("Warning: Source dataset has 0 elements!")
        if len(self.target_dataset) == 0:
            print("Warning: Target dataset has 0 elements!")
            
    def __len__(self):
        # The epoch length is defined by the larger source dataset
        return len(self.source_dataset)
        
    def __getitem__(self, idx):
        # 1. Get source sample
        source_sample = self.source_dataset[idx]
        
        # Unpack source sample (depends on whether return_ft is enabled)
        if self.source_dataset.return_ft:
            source_img, source_ft, source_label = source_sample
        else:
            source_img, source_label = source_sample
            source_ft = None
            
        # 2. Get target sample (using modulo-based index mapping)
        if len(self.target_dataset) > 0:
            target_idx = idx % len(self.target_dataset)
            target_sample = self.target_dataset[target_idx]
            
            if self.target_dataset.return_ft:
                target_img, target_ft, target_label = target_sample
            else:
                target_img, target_label = target_sample
                target_ft = None
        else:
            # Fallback if target dataset is empty
            target_img = torch.zeros_like(source_img)
            target_label = -1
            target_ft = torch.zeros_like(source_ft) if source_ft is not None else None
            
        # 3. Construct outputs dictionary
        batch_dict = {
            "source_img": source_img,
            "source_label": source_label,
            "target_img": target_img,
            "target_label": target_label
        }
        
        if source_ft is not None:
            batch_dict["source_ft"] = source_ft
        if target_ft is not None:
            batch_dict["target_ft"] = target_ft
            
        return batch_dict


def get_target_dataloader(target_dir, split, batch_size, input_size, csv_path=None, use_fourier=False, is_train=True, num_workers=4):
    """
    Creates a dataloader for target domain dataset alone (useful for evaluation/validation).
    """
    if is_train:
        # Replicate training augmentations
        color_transform = T.Compose([
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2)
        ])
        
        transform = T.Compose([
            T.ToPILImage(),
            T.RandomCrop(size=(input_size, input_size), pad_if_needed=True, padding_mode="reflect"),
            RandomRotationWithReflect(15),
            T.RandomHorizontalFlip(),
            color_transform,
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = T.Compose([
            T.ToPILImage(),
            T.CenterCrop(input_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    k_size = (input_size + 15) // 16
    fourier_size = (k_size * 2, k_size * 2)
    
    target_dataset = TargetDomainDataset(
        root_dir=target_dir,
        split=split,
        csv_path=csv_path,
        transform=transform,
        return_ft=use_fourier,
        fourier_size=fourier_size
    )
    
    dataloader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=is_train,
        pin_memory=True,
        num_workers=num_workers if len(target_dataset) > 0 else 0,
        persistent_workers=True if (num_workers > 0 and len(target_dataset) > 0) else False
    )
    return dataloader


def get_domain_balanced_dataloader(
    source_dir, 
    target_dir, 
    split, 
    batch_size, 
    input_size, 
    target_csv_path=None,
    use_fourier=False, 
    is_train=True, 
    num_workers=4, 
    use_randaugment=False, 
    ra_num_ops=2, 
    ra_magnitude=9
):
    """
    Creates a balanced dataloader containing both source domain and target domain samples.
    """
    # 1. Define Transforms aligned with dataset.py (commit 216962e0d4fc5ca194f88912b19269a6dc6f35e5)
    if is_train:
        if use_randaugment:
            color_transform = SafeRandAugment(num_ops=ra_num_ops, magnitude=ra_magnitude)
        else:
            color_transform = T.Compose([
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
                T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2)
            ])
            
        transform_train = T.Compose([
            T.ToPILImage(),
            T.RandomCrop(size=(input_size, input_size), pad_if_needed=True, padding_mode="reflect"),
            RandomRotationWithReflect(15),
            T.RandomHorizontalFlip(),
            color_transform,
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        source_transform = (transform_train, transform_train)
        target_transform = transform_train
    else:
        transform_val = T.Compose([
            T.ToPILImage(),
            T.CenterCrop(input_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        source_transform = transform_val
        target_transform = transform_val
        
    k_size = (input_size + 15) // 16
    fourier_size = (k_size * 2, k_size * 2)
    
    # 2. Instantiate datasets
    source_dataset = AntiSpoofingDataset(
        root_dir=source_dir,
        split=split,
        transform=source_transform,
        return_ft=use_fourier,
        fourier_size=fourier_size
    )
    
    target_dataset = TargetDomainDataset(
        root_dir=target_dir,
        split=split,
        csv_path=target_csv_path,
        transform=target_transform,
        return_ft=use_fourier,
        fourier_size=fourier_size
    )
    
    # 3. Wrap in DomainBalancedDataset
    balanced_dataset = DomainBalancedDataset(
        source_dataset=source_dataset,
        target_dataset=target_dataset
    )
    
    # 4. Create DataLoader
    dataloader = DataLoader(
        balanced_dataset,
        batch_size=batch_size,
        shuffle=is_train,
        pin_memory=True,
        num_workers=num_workers if len(balanced_dataset) > 0 else 0,
        persistent_workers=True if (num_workers > 0 and len(balanced_dataset) > 0) else False
    )
    
    return dataloader
