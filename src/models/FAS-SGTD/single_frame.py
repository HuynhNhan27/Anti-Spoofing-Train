"""
PyTorch implementation of LstmCnnNet model
Converted from TensorFlow version in generate_network.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpatialGradientLayer(nn.Module):
    """Compute spatial gradients using Sobel operators"""
    
    def __init__(self):
        super(SpatialGradientLayer, self).__init__()
        
        # Sobel kernel for X direction
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        self.register_buffer('sobel_x', torch.from_numpy(sobel_x))
        
        # Sobel kernel for Y direction
        sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
        self.register_buffer('sobel_y', torch.from_numpy(sobel_y))
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) input tensor
        Returns:
            gradient_x: spatial gradient in X direction
            gradient_y: spatial gradient in Y direction
        """
        B, C, H, W = x.shape
        
        # Create depthwise Sobel kernels for each channel
        sobel_x_kernel = self.sobel_x.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)
        sobel_y_kernel = self.sobel_y.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)
        
        # Apply depthwise convolution (groups=C means one kernel per channel)
        gradient_x = F.conv2d(x, sobel_x_kernel.to(x.device), padding=1, groups=C)
        gradient_y = F.conv2d(x, sobel_y_kernel.to(x.device), padding=1, groups=C)
        
        return gradient_x, gradient_y


class ResidualGradientConv(nn.Module):
    """Residual Gradient Convolution Block"""
    
    def __init__(self, in_channels, out_channels, is_training=True, 
                 gradient_type='type1', use_batch_norm=True):
        super(ResidualGradientConv, self).__init__()
        
        self.gradient_type = gradient_type
        self.use_batch_norm = use_batch_norm
        self.is_training = is_training
        
        # 3x3 convolution
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                             stride=1, padding=1, bias=False)
        
        # 1x1 convolution for gradient (Gabor filter)
        self.conv_gabor = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                   stride=1, padding=0, bias=False)
        
        # Batch normalization layers
        if use_batch_norm:
            self.bn_main = nn.BatchNorm2d(out_channels)
            self.bn_gabor = nn.BatchNorm2d(out_channels)
        
        # Spatial gradient layer
        self.spatial_gradient = SpatialGradientLayer()
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) input tensor
        Returns:
            output: (B, out_channels, H, W)
        """
        # Main convolution path
        out = self.conv(x)
        
        # Compute spatial gradients
        grad_x, grad_y = self.spatial_gradient(x)
        
        # Compute gradient magnitude based on type
        if self.gradient_type == 'type0':
            # L2 norm squared
            grad_mag = torch.pow(grad_x, 2) + torch.pow(grad_y, 2)
        elif self.gradient_type == 'type1':
            # L2 norm
            grad_mag = torch.sqrt(torch.pow(grad_x, 2) + torch.pow(grad_y, 2) + 1e-8)
        elif self.gradient_type == 'type2':
            # No gradient path
            grad_mag = None
        else:
            raise ValueError(f"Unknown gradient_type: {self.gradient_type}")
        
        # Add gradient path
        if grad_mag is not None:
            grad_processed = self.conv_gabor(grad_mag)
            if self.use_batch_norm:
                grad_processed = self.bn_gabor(grad_processed)
            out = out + grad_processed
        
        # Batch normalization on main path
        if self.use_batch_norm:
            out = self.bn_main(out)
        
        # Activation
        out = self.relu(out)
        
        return out


class FaceMapNet(nn.Module):
    """Face Anti-Spoofing Map Network - Generates depth/liveness maps"""
    
    def __init__(self, len_seq=1, multiplier=2):
        super(FaceMapNet, self).__init__()
        
        self.len_seq = len_seq
        self.multiplier = multiplier
        
        # Initial residual gradient conv
        self.rgc_init = ResidualGradientConv(3, 64)
        
        # Block 1
        self.rgc1_1 = ResidualGradientConv(64, 64 * multiplier)
        self.rgc1_2 = ResidualGradientConv(64 * multiplier, 96 * multiplier)
        self.rgc1_3 = ResidualGradientConv(96 * multiplier, 64 * multiplier)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2
        self.rgc2_1 = ResidualGradientConv(64 * multiplier, 64 * multiplier)
        self.rgc2_2 = ResidualGradientConv(64 * multiplier, 96 * multiplier)
        self.rgc2_3 = ResidualGradientConv(96 * multiplier, 64 * multiplier)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3
        self.rgc3_1 = ResidualGradientConv(64 * multiplier, 64 * multiplier)
        self.rgc3_2 = ResidualGradientConv(64 * multiplier, 96 * multiplier)
        self.rgc3_3 = ResidualGradientConv(96 * multiplier, 64 * multiplier)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 4 (multi-scale fusion)
        concat_channels = 64 * multiplier + 64 * multiplier + 64 * multiplier  # pool1 + pool2 + pool3
        self.rgc4_1 = ResidualGradientConv(concat_channels, 64 * multiplier)
        self.rgc4_2 = ResidualGradientConv(64 * multiplier, 32 * multiplier)
        
        # Final output conv
        self.conv_out = nn.Conv2d(32 * multiplier, len_seq, kernel_size=3, 
                                 stride=1, padding=1)
        self.relu_out = nn.ReLU()
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input image tensor (values in [0, 255])
        Returns:
            logits_map: (B, len_seq, H, W) output map
        """
        # Normalize input if needed (assuming input is [0, 255])
        # You may need to adjust this based on your input
        
        # Initial conv
        feat = self.rgc_init(x)
        
        # Block 1
        feat = self.rgc1_1(feat)
        feat = self.rgc1_2(feat)
        feat = self.rgc1_3(feat)
        pool1 = self.pool1(feat)
        
        # Block 2
        feat = self.rgc2_1(pool1)
        feat = self.rgc2_2(feat)
        feat = self.rgc2_3(feat)
        pool2 = self.pool2(feat)
        
        # Block 3
        feat = self.rgc3_1(pool2)
        feat = self.rgc3_2(feat)
        feat = self.rgc3_3(feat)
        pool3 = self.pool3(feat)
        
        # Multi-scale feature fusion
        # Resize pool1 and pool2 to match pool3 spatial dimensions, then concat
        feat1_resized = F.interpolate(pool1, size=pool3.shape[-2:], mode='bilinear', align_corners=False)
        feat2_resized = F.interpolate(pool2, size=pool3.shape[-2:], mode='bilinear', align_corners=False)
        
        pool_concat = torch.cat([feat1_resized, feat2_resized, pool3], dim=1)
        
        # Block 4 with concatenated features
        feat = self.rgc4_1(pool_concat)
        feat = self.rgc4_2(feat)
        
        # Output map
        logits_map = self.conv_out(feat)
        logits_map = self.relu_out(logits_map)
        
        return logits_map


class SoftmaxNet(nn.Module):
    """Classification head for computing logits"""
    
    def __init__(self, input_features=1024, num_classes=2):
        super(SoftmaxNet, self).__init__()
        
        self.fc1 = nn.Linear(input_features, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, -1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x


class LstmCnnNet(nn.Module):
    """Main LSTM+CNN Network for Face Anti-Spoofing"""
    
    def __init__(self, len_seq=1, num_classes=2, multiplier=2, input_features=1024):
        super(LstmCnnNet, self).__init__()
        
        self.len_seq = len_seq
        self.num_classes = num_classes
        
        # FaceMapNet for depth map generation
        self.face_map_net = FaceMapNet(len_seq=len_seq, multiplier=multiplier)
        
        # Classification head
        self.softmax_net = SoftmaxNet(input_features=input_features, num_classes=num_classes)
    
    def forward(self, x, is_training=True):
        """
        Args:
            x: (B, seq_len, C, H, W) or (B, C, H, W) input image/sequence
            is_training: bool, whether in training mode (checked via self.training)
        Returns:
            If training: (logits_cla, logits_map)
            If eval: logits_cla
        """
        # If input is 5D (batch, seq, C, H, W), take first frame or process accordingly
        if len(x.shape) == 5:
            x = x[:, 0, :, :, :]  # Take first frame of sequence
        
        logits_map = self.face_map_net(x)
        
        # Pass depth map to classification head
        logits_cla = self.softmax_net(logits_map)
        
        if self.training:
            return logits_cla, logits_map
        else:
            return logits_cla


# Utility functions for loss computation

def contrast_depth_conv(x, dilation_rate=1):
    """
    Compute contrast depth using directional filters
    
    Args:
        x: (B, 1, H, W) input tensor
    Returns:
        contrast_depth: (B, 8, H, W) tensor with 8 directional contrasts
    """
    kernel_list = [
        [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
        [[0, 1, 0], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 1], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 0], [1, -1, 0], [0, 0, 0]],
        [[0, 0, 0], [0, -1, 1], [0, 0, 0]],
        [[0, 0, 0], [0, -1, 0], [1, 0, 0]],
        [[0, 0, 0], [0, -1, 0], [0, 1, 0]],
        [[0, 0, 0], [0, -1, 0], [0, 0, 1]]
    ]
    
    kernel = np.array(kernel_list, dtype=np.float32)  # (8, 3, 3)
    kernel = torch.from_numpy(kernel).unsqueeze(1)  # (8, 1, 3, 3)
    kernel = kernel.to(x.device)
    
    if dilation_rate == 1:
        contrast_depth = F.conv2d(x, kernel, padding=1)
    else:
        contrast_depth = F.conv2d(x, kernel, padding=dilation_rate, dilation=dilation_rate)
    
    return contrast_depth


def contrast_depth_loss(output, target):
    """
    Compute contrast depth loss
    
    Args:
        output: (B, 1, H, W) predicted depth map
        target: (B, 1, H, W) target depth map
    Returns:
        loss: scalar tensor
    """
    contrast_out = contrast_depth_conv(output, dilation_rate=1)
    contrast_target = contrast_depth_conv(target, dilation_rate=1)
    
    loss = torch.pow(contrast_out - contrast_target, 2)
    loss = torch.mean(loss)
    
    return loss


# Example usage
if __name__ == "__main__":
    # Create model
    model = LstmCnnNet(len_seq=1, num_classes=2)
    model.eval()
    
    # Create dummy input (batch_size=4, channels=3, height=256, width=256)
    dummy_input = torch.randn(4, 3, 256, 256)
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input, is_training=False)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Model conversion successful!")
