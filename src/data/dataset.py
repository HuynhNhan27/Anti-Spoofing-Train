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
    Tùy chỉnh RandAugment cho Face Anti-Spoofing (Dựa trên Train-EDA cân bằng).
    - Loại bỏ các phép toán băm nát texture (Posterize, Solarize, Equalize).
    - Áp dụng nhiễu loạn đối xứng (Symmetric Jitter) nhẹ nhàng cho CẢ 2 nhãn.
    - Không làm mờ quá tay để giữ nguyên phân phối Sharpness gốc.
    """
    def __init__(self, num_ops=2, magnitude=9):
        self.num_ops = num_ops
        # Giảm scale của magnitude xuống để các phép biến đổi "nhẹ tay" hơn
        self.magnitude = magnitude
        
    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = T.ToPILImage()(img)
            
        # Hệ số tác động (tối đa ~0.3 tức là thay đổi +- 30%)
        mag_factor = self.magnitude / 30.0 
        
        operations = [
            lambda x: F.autocontrast(x),
            # Contrast: Nhiễu loạn +- 30%
            lambda x: F.adjust_contrast(x, 1.0 + mag_factor * random.choice([-1, 1])),
            # Brightness: Nhiễu loạn +- 30%
            lambda x: F.adjust_brightness(x, 1.0 + mag_factor * random.choice([-1, 1])),
            # Saturation: Nhiễu loạn +- 30% (Vì 2 class đang khá sát nhau ở mốc 60-55)
            lambda x: F.adjust_saturation(x, 1.0 + mag_factor * random.choice([-1, 1])),
            # Hue: Nhiễu cực nhẹ (+- 5%) để giữ lại đặc trưng màu màn hình/mực in
            lambda x: F.adjust_hue(x, (mag_factor * 0.15) * random.choice([-1, 1])),
            # Sharpness: Chủ yếu là làm mờ đi một chút hoặc giữ nguyên, hiếm khi làm nét
            lambda x: F.adjust_sharpness(x, max(0.2, 1.0 - mag_factor * 1.5 * random.random())), 
            # Gaussian Blur: Kernel nhỏ (3) để thỉnh thoảng tạo độ rung mờ nhẹ
            lambda x: F.gaussian_blur(x, kernel_size=3, sigma=0.1 + mag_factor)
        ]
            
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
        # Không dùng nén/phóng ảnh (no resizing/resized crops)
        
        if use_randaugment:
            # Dùng chung một cấu hình SafeRandAugment cho cả 2 nhãn (Symmetric)
            # Lưu ý: Cần đảm bảo class SafeRandAugment đã được cập nhật (bỏ logic is_spoof)
            color_transform = SafeRandAugment(num_ops=ra_num_ops, magnitude=ra_magnitude)
        else:
            color_transform = T.Compose([
                # Nhiễu loạn đối xứng: Dao động +- 30% cho cả 2 nhãn, Hue thay đổi nhẹ +- 5%
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
                # Làm mờ nhẹ ngẫu nhiên (xác suất 20%) với kernel nhỏ cho cả 2 nhãn
                T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2)
            ])
            
        # Một Pipeline duy nhất cho tập Train
        transform_train = T.Compose([
            T.ToPILImage(),
            # Thêm lại pad_if_needed=True để tránh lỗi nếu ảnh có chiều < 224 (VD: 160x240)
            T.RandomCrop(size=(224, 224), pad_if_needed=True, padding_mode="reflect"),
            RandomRotationWithReflect(15),
            T.RandomHorizontalFlip(),
            color_transform, # Áp dụng màu sắc/độ nét chung
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Nếu class Dataset của bạn được thiết kế để nhận tuple (transform_live, transform_spoof)
        # Ta gán transform_train cho cả hai để giữ nguyên cấu trúc code cũ mà không gây lỗi
        transform = (transform_train, transform_train)  
    else:
        transform = T.Compose([
            T.ToPILImage(),
            # Đảm bảo Val/Test cũng có Pad nếu ảnh nhỏ hơn 224 để CenterCrop không văng lỗi
            # Tùy thuộc vào việc bạn viết custom SquarePad như trước hay dùng mặc định của PyTorch
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
