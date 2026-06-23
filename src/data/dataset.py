import os
import cv2
import math
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as F

# Class mapping: Real/Live is 0, Spoof/Fake is 1
CLASS_MAP = {
    "live": 0,
    "spoof": 1
}

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def generate_FT(image):
    """
    Generate Fourier Transform map of the image.
    Expects image in RGB format (numpy array).
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray_image)
    fshift = np.fft.fftshift(f)
    fimg = np.log(np.abs(fshift) + 1)
    
    # Normalize to [0, 1]
    fimg_min = fimg.min()
    fimg_max = fimg.max()
    if fimg_max > fimg_min:
        fimg = (fimg - fimg_min) / (fimg_max - fimg_min)
    else:
        fimg = np.zeros_like(fimg)
    return fimg

class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [
            max_wh - (s + pad) for s, pad in zip(image.size, [p_left, p_top])
        ]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 0, "constant")

class RandomRotationWithReflect:
    """
    Randomly rotate the image and fill background pixels using boundary reflection (BORDER_REFLECT_101).
    Very useful in face anti-spoofing to avoid artificial black corners.
    """
    def __init__(self, degrees, expand=False):
        self.degrees = degrees
        self.expand = expand

    def __call__(self, img):
        angle = T.RandomRotation.get_params([-self.degrees, self.degrees])

        if isinstance(img, Image.Image):
            img_np = np.array(img, dtype=np.uint8)
        else:
            img_np = np.array(img, dtype=np.uint8)

        h, w = img_np.shape[:2]
        center = (w // 2, h // 2)

        if self.expand:
            cos = np.abs(math.cos(math.radians(angle)))
            sin = np.abs(math.sin(math.radians(angle)))
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))

            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]

            img_rotated = cv2.warpAffine(
                img_np,
                M,
                (new_w, new_h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            return Image.fromarray(img_rotated, "RGB")
        else:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img_rotated = cv2.warpAffine(
                img_np,
                M,
                (w, h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            return Image.fromarray(img_rotated, "RGB")

class AntiSpoofingDataset(Dataset):
    """
    Unified Dataset for Face Anti-Spoofing.
    Loads images from standard Folder structure:
    data_dir/split/live/  -> Label 0
    data_dir/split/spoof/ -> Label 1
    """
    def __init__(self, root_dir, split="train", transform=None, return_ft=False, fourier_size=(16, 16)):
        self.split_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.return_ft = return_ft
        self.fourier_size = fourier_size
        
        self.samples = []
        
        # Scan directories
        for class_name, label in CLASS_MAP.items():
            class_dir = os.path.join(self.split_dir, class_name)
            if not os.path.exists(class_dir):
                # Optionally warn, but continue if split directory doesn't have it
                continue
                
            for root, _, files in os.walk(class_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        img_path = os.path.join(root, file)
                        self.samples.append((img_path, label))
                        
        print(f"Loaded {len(self.samples)} images for split '{split}' from {self.split_dir}")

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

def get_dataloader(data_dir, split, batch_size, input_size, use_fourier=False, is_train=True, num_workers=4):
    """
    Factory function to get data loader with correct augmentations.
    """
    if is_train:
        transform = T.Compose([
            T.ToPILImage(),
            T.RandomResizedCrop(size=(input_size, input_size), scale=(0.9, 1.1)),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            RandomRotationWithReflect(90),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])
    else:
        transform = T.Compose([
            T.ToPILImage(),
            SquarePad(),
            T.Resize((input_size, input_size)),
            T.ToTensor(),
        ])
        
    # Calculate fourier size: input_size=128 -> (16, 16)
    k_size = (input_size + 15) // 16
    fourier_size = (k_size * 2, k_size * 2)
    
    dataset = AntiSpoofingDataset(
        root_dir=data_dir,
        split=split,
        transform=transform,
        return_ft=use_fourier,
        fourier_size=fourier_size
    )
    
    # Handle empty dataset gracefully
    if len(dataset) == 0:
        print(f"Warning: Empty dataset found for split: {split}")
        
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        pin_memory=True,
        num_workers=num_workers if len(dataset) > 0 else 0
    )
    return dataloader
