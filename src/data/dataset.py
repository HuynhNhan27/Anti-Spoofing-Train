import os
import cv2
import random
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
                        
        # Subsample validation dataset to randomly select only 30% of validation images
        if split == "val":
            random_state = random.getstate()
            random.seed(42)
            num_samples = int(len(self.samples) * 0.3)
            self.samples = random.sample(self.samples, num_samples)
            random.setstate(random_state)
            
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
            if isinstance(self.transform, (tuple, list)):
                # Apply class-specific transform (0: live, 1: spoof)
                trans = self.transform[0] if label == 0 else self.transform[1]
                if trans is not None:
                    image = trans(image)
            else:
                image = self.transform(image)
            
        if self.return_ft:
            return image, ft_sample, label
        return image, label

class SafeRandAugment:
    """
    Tùy chỉnh RandAugment cho Face Anti-Spoofing.
    Loại bỏ các phép biến đổi phá hủy texture (Posterize, Solarize, Equalize).
    Áp dụng điều chỉnh định hướng để khớp phân phối EDA (Live ép giảm màu/độ nét, Spoof ép tăng).
    """
    def __init__(self, num_ops=2, magnitude=9, is_spoof=False):
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.is_spoof = is_spoof
        
    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = T.ToPILImage()(img)
            
        magnitude_factor = self.magnitude / 30.0
        
        # CÁC PHÉP TOÁN CHUNG (Thay đổi nhẹ ánh sáng hai chiều)
        operations = [
            lambda x: F.autocontrast(x),
            lambda x: F.adjust_brightness(x, 1.0 + magnitude_factor * 0.3 * random.choice([-1, 1])),
            lambda x: F.adjust_hue(x, magnitude_factor * 0.1 * random.choice([-1, 1]))
        ]
        
        if self.is_spoof:
            # === LOGIC CHO SPOOF ===
            # Spoof nhạt màu và kém tương phản -> ÉP TĂNG 
            operations.extend([
                lambda x: F.adjust_contrast(x, 1.0 + magnitude_factor * 0.4), # Chỉ tăng > 1.0
                lambda x: F.adjust_saturation(x, 1.0 + magnitude_factor * 0.5), # Chỉ tăng > 1.0
            ])
            # Tuyệt đối không thêm Blur hay Sharpness vào Spoof
        else:
            # === LOGIC CHO LIVE ===
            # Live màu rực rỡ và rất sắc nét -> ÉP GIẢM
            operations.extend([
                lambda x: F.adjust_contrast(x, max(0.5, 1.0 - magnitude_factor * 0.4)), # Chỉ giảm < 1.0
                lambda x: F.adjust_saturation(x, max(0.4, 1.0 - magnitude_factor * 0.6)), # Chỉ giảm < 1.0
                # Factor < 1.0 trong adjust_sharpness tương đương với làm mờ đi
                lambda x: F.adjust_sharpness(x, max(0.1, 1.0 - magnitude_factor * 0.8)), 
                # Thêm Gaussian Blur với kernel ngẫu nhiên
                lambda x: F.gaussian_blur(x, kernel_size=random.choice([3, 5, 7]), sigma=0.5 + magnitude_factor)
            ])
            
        # Randomly apply operations
        ops = random.sample(operations, min(self.num_ops, len(operations)))
        for op in ops:
            img = op(img)
        return img

def get_dataloader(data_dir, split, batch_size, input_size, use_fourier=False, is_train=True, num_workers=4, use_randaugment=False, ra_num_ops=2, ra_magnitude=9):
    """
    Factory function to get data loader with correct augmentations.
    """
    if is_train:
        # Patch-based training (crop input_size x input_size, pad if needed)
        # Hạn chế biến đổi hình học (rotate max 15, horizontal flip)
        # Thêm làm mờ nhẹ và color jitter ở ảnh live
        # Không dùng nén/phóng ảnh (no resizing/resized crops)
        if use_randaugment:
            live_color_transform = SafeRandAugment(num_ops=ra_num_ops, magnitude=ra_magnitude, is_spoof=False)
            spoof_color_transform = SafeRandAugment(num_ops=ra_num_ops, magnitude=ra_magnitude, is_spoof=True)
        else:
            live_color_transform = T.Compose([
                # Ép Saturation CHỈ GIẢM (0.4 đến 1.0), giữ nguyên hoặc giảm Contrast
                T.ColorJitter(brightness=0.2, contrast=(0.6, 1.0), saturation=(0.4, 1.0), hue=0.05),
                # Tăng kernel_size lên một chút (5, 7) để đảm bảo đủ mờ
                T.RandomApply([T.GaussianBlur(kernel_size=(5, 7), sigma=(0.5, 1.5))], p=0.5)
            ])
            
            spoof_color_transform = T.Compose([
                # Ép Saturation CHỈ TĂNG (1.0 đến 1.4), giữ nguyên hoặc tăng Contrast
                T.ColorJitter(brightness=0.2, contrast=(1.0, 1.3), saturation=(1.0, 1.4), hue=0.05),
            ])
            
        transform_live = T.Compose([
            T.ToPILImage(),
            T.RandomCrop(size=(224, 224)),
            RandomRotationWithReflect(15),
            T.RandomHorizontalFlip(),
            live_color_transform,
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        spoof_transforms = [
            T.ToPILImage(),
            T.RandomCrop(size=(224, 224)),
            RandomRotationWithReflect(15),
            T.RandomHorizontalFlip(),
        ]
        if spoof_color_transform is not None:
            spoof_transforms.append(spoof_color_transform)
        spoof_transforms.append(T.ToTensor())
        spoof_transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        
        transform_spoof = T.Compose(spoof_transforms)
        
        transform = (transform_live, transform_spoof)
    else:
        transform = T.Compose([
            T.ToPILImage(),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    # Calculate fourier size based on the final image size (224x224)
    k_size = (224 + 15) // 16
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
        num_workers=num_workers if len(dataset) > 0 else 0,
        persistent_workers=True if (num_workers > 0 and len(dataset) > 0) else False
    )
    return dataloader