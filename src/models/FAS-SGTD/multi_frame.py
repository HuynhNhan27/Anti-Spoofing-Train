"""
PyTorch implementation of LstmCnnNet model for multi-frame face anti-spoofing
Converted from TensorFlow version in fas_sgtd_multi_frame/generate_network.py

Architecture:
1. FaceMapNet: Extract depth maps and intermediate features
2. OFFNet: Compute Optical Flow Features with temporal gradients
3. ConvGRUNet: Process temporal sequence with Convolutional GRU
4. Fusion: Combine single-frame and temporal results
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
            grad_mag = torch.pow(grad_x, 2) + torch.pow(grad_y, 2)
        elif self.gradient_type == 'type1':
            grad_mag = torch.sqrt(torch.pow(grad_x, 2) + torch.pow(grad_y, 2) + 1e-8)
        elif self.gradient_type == 'type2':
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
    """Face Anti-Spoofing Map Network - Generates depth maps and intermediate features"""
    
    def __init__(self, multiplier=2):
        super(FaceMapNet, self).__init__()
        
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
        concat_channels = 64 * multiplier + 64 * multiplier + 64 * multiplier
        self.rgc4_1 = ResidualGradientConv(concat_channels, 64 * multiplier)
        self.rgc4_2 = ResidualGradientConv(64 * multiplier, 32 * multiplier)
        
        # Final output conv
        self.conv_out = nn.Conv2d(32 * multiplier, 1, kernel_size=3, 
                                 stride=1, padding=1)
        self.relu_out = nn.ReLU()
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input image tensor
        Returns:
            features_map: (B, 1, H, W) output depth map
            pre_off_list: list of 3 intermediate features for OFFNet
        """
        pre_off_list = []
        
        # Initial conv
        feat = self.rgc_init(x)
        
        # Block 1
        feat = self.rgc1_1(feat)
        feat = self.rgc1_2(feat)
        feat = self.rgc1_3(feat)
        pre_off_list.append(feat)  # Store intermediate feature
        pool1 = self.pool1(feat)
        
        # Block 2
        feat = self.rgc2_1(pool1)
        feat = self.rgc2_2(feat)
        feat = self.rgc2_3(feat)
        pre_off_list.append(feat)  # Store intermediate feature
        pool2 = self.pool2(feat)
        
        # Block 3
        feat = self.rgc3_1(pool2)
        feat = self.rgc3_2(feat)
        feat = self.rgc3_3(feat)
        pre_off_list.append(feat)  # Store intermediate feature
        pool3 = self.pool3(feat)
        
        # Multi-scale feature fusion
        feat1_resized = F.interpolate(pool1, size=pool3.shape[-2:], mode='bilinear', align_corners=False)
        feat2_resized = F.interpolate(pool2, size=pool3.shape[-2:], mode='bilinear', align_corners=False)
        pool_concat = torch.cat([feat1_resized, feat2_resized, pool3], dim=1)
        
        # Block 4 with concatenated features
        feat = self.rgc4_1(pool_concat)
        feat = self.rgc4_2(feat)
        
        # Output map
        features_map = self.conv_out(feat)
        features_map = self.relu_out(features_map)
        
        return features_map, pre_off_list


class ConvGRUCell(nn.Module):
    """Convolutional GRU Cell for processing spatial-temporal features"""
    
    def __init__(self, input_channels, hidden_channels, kernel_size=3, activation=torch.tanh, 
                 last_activation=None):
        super(ConvGRUCell, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.last_activation = last_activation
        
        self.padding = kernel_size // 2
        
        # Gates: reset and update
        self.conv_gates = nn.Conv2d(
            input_channels + hidden_channels,
            2 * hidden_channels,
            kernel_size,
            padding=self.padding
        )
        
        # Candidate
        self.conv_candidate = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size,
            padding=self.padding
        )
    
    def forward(self, x, h):
        """
        Args:
            x: (B, C_in, H, W) input tensor
            h: (B, C_hidden, H, W) hidden state
        Returns:
            h_new: (B, C_hidden, H, W) updated hidden state
        """
        combined = torch.cat([x, h], dim=1)
        
        # Compute reset and update gates
        gates = self.conv_gates(combined)
        reset_gate, update_gate = torch.split(gates, self.hidden_channels, dim=1)
        reset_gate = torch.sigmoid(reset_gate)
        update_gate = torch.sigmoid(update_gate)
        
        # Compute candidate hidden state
        combined_candidate = torch.cat([x, reset_gate * h], dim=1)
        candidate = self.conv_candidate(combined_candidate)
        candidate = self.activation(candidate)
        
        # Compute new hidden state
        h_new = (1 - update_gate) * candidate + update_gate * h
        
        if self.last_activation is not None:
            h_new = self.last_activation(h_new)
        
        return h_new


class ConvGRUNet(nn.Module):
    """Convolutional GRU Network for processing temporal sequences"""
    
    def __init__(self, input_channels, len_seq):
        super(ConvGRUNet, self).__init__()
        
        self.input_channels = input_channels
        self.len_seq = len_seq
        
        # First GRU cell
        self.gru_cell_1 = ConvGRUCell(input_channels, 64, kernel_size=3)
        
        # Second GRU cell (output channels = 1)
        self.gru_cell_2 = ConvGRUCell(64, 1, kernel_size=3, last_activation=None)
        
        # Third GRU cell (output channels = 1 with tanh activation)
        self.gru_cell_3 = ConvGRUCell(1, 1, kernel_size=3, last_activation=torch.tanh)
    
    def forward(self, input_sequence):
        """
        Args:
            input_sequence: (T, B, C, H, W) where T = len_seq-1 (time steps)
        Returns:
            depth_maps: list of (B, 1, H, W) depth maps for each time step
        """
        T, B, C, H, W = input_sequence.shape
        
        # Initialize hidden states
        h1 = torch.zeros(B, 64, H, W, device=input_sequence.device, dtype=input_sequence.dtype)
        h2 = torch.zeros(B, 1, H, W, device=input_sequence.device, dtype=input_sequence.dtype)
        h3 = torch.zeros(B, 1, H, W, device=input_sequence.device, dtype=input_sequence.dtype)
        
        depth_maps = []
        
        # Process each time step
        for t in range(T):
            x = input_sequence[t]  # (B, C, H, W)
            
            # First GRU layer
            h1 = self.gru_cell_1(x, h1)
            
            # Second GRU layer
            h2 = self.gru_cell_2(h1, h2)
            
            # Third GRU layer
            h3 = self.gru_cell_3(h2, h3)
            
            depth_maps.append(h3)
        
        return depth_maps


class OFFNet(nn.Module):
    """Optical Flow Features Network - Computes temporal and spatial gradients"""
    
    def __init__(self, len_seq, multiplier=2, reduce_num=32):
        super(OFFNet, self).__init__()
        
        self.len_seq = len_seq
        self.reduce_num = reduce_num
        in_channels = 64 * multiplier
        
        # Sobel kernels (will be created dynamically based on input channels)
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        self.register_buffer('sobel_x_base', torch.from_numpy(sobel_x))
        
        sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
        self.register_buffer('sobel_y_base', torch.from_numpy(sobel_y))
        
        # Feature reduction layers for each block
        self.conv_reduce_1 = nn.Conv2d(in_channels, reduce_num, kernel_size=1, stride=1)
        self.bn_reduce_1 = nn.BatchNorm2d(reduce_num)
        self.conv_out_1 = nn.Conv2d(reduce_num * 6, in_channels, kernel_size=3, stride=1, padding=1)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv_reduce_2 = nn.Conv2d(in_channels, reduce_num, kernel_size=1, stride=1)
        self.bn_reduce_2 = nn.BatchNorm2d(reduce_num)
        self.conv_out_2 = nn.Conv2d(reduce_num * 6, in_channels, kernel_size=3, stride=1, padding=1)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv_reduce_3 = nn.Conv2d(in_channels, reduce_num, kernel_size=1, stride=1)
        self.bn_reduce_3 = nn.BatchNorm2d(reduce_num)
        self.conv_out_3 = nn.Conv2d(reduce_num * 6, in_channels, kernel_size=3, stride=1, padding=1)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Cascade fusion layers
        self.conv_cascade_1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn_cascade_1 = nn.BatchNorm2d(in_channels)
        
        self.conv_cascade_2 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn_cascade_2 = nn.BatchNorm2d(in_channels)
        
        # Final layer
        self.conv_final = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn_final = nn.BatchNorm2d(in_channels)
        
        self.relu = nn.ReLU(inplace=True)
    
    def compute_off_features(self, pre_off_feature, reduce_layer, bn_layer, out_layer, 
                            pool_layer, reduce_num):
        """
        Compute OFF features from intermediate feature maps
        
        Args:
            pre_off_feature: (B*T, C, H, W)
            reduce_num: number of channels for reduction
        Returns:
            res_feature: (B*(T-1), C, H/2, W/2)
        """
        B_T, C, H, W = pre_off_feature.shape
        B = B_T // self.len_seq
        
        # Reshape to (B, T, C, H, W)
        feature_seq = pre_off_feature.view(B, self.len_seq, C, H, W)
        
        # Reduce channels
        net = reduce_layer(pre_off_feature)
        net = bn_layer(net)
        net = self.relu(net)
        
        # Reshape for temporal processing
        net_seq = net.view(B, self.len_seq, self.reduce_num, H, W)
        
        # Compute spatial gradients
        grad_x, grad_y = self._compute_spatial_gradients(net, self.reduce_num)
        grad_x_seq = grad_x.view(B, self.len_seq, self.reduce_num, H, W)
        grad_y_seq = grad_y.view(B, self.len_seq, self.reduce_num, H, W)
        
        # Compute temporal gradients
        temporal_grad = net_seq[:, :-1, :, :, :] - net_seq[:, 1:, :, :, :]
        
        # Concatenate OFF features
        off_concat = torch.cat([
            net_seq[:, :-1, :, :, :],
            grad_x_seq[:, :-1, :, :, :],
            grad_y_seq[:, :-1, :, :, :],
            grad_x_seq[:, 1:, :, :, :],
            grad_y_seq[:, 1:, :, :, :],
            temporal_grad
        ], dim=2)  # (B, T-1, 6*reduce_num, H, W)
        
        # Reshape back to batch format
        off_batch = off_concat.view(-1, off_concat.shape[2], H, W)
        
        # Output convolution
        res_feature = out_layer(off_batch)
        res_feature = self.relu(res_feature)
        res_feature = pool_layer(res_feature)
        
        return res_feature
    
    def _compute_spatial_gradients(self, x, channels):
        """Compute Sobel gradients for each channel"""
        sobel_x = self.sobel_x_base.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
        sobel_y = self.sobel_y_base.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
        
        grad_x = F.conv2d(x, sobel_x.to(x.device), padding=1, groups=channels)
        grad_y = F.conv2d(x, sobel_y.to(x.device), padding=1, groups=channels)
        
        return grad_x, grad_y
    
    def forward(self, pre_off_list):
        """
        Args:
            pre_off_list: list of 3 intermediate features, each (B*T, C, H, W)
        Returns:
            net: (B*(T-1), 128, 32, 32) output features
        """
        # Process each OFF block
        net1 = self.compute_off_features(
            pre_off_list[0], self.conv_reduce_1, self.bn_reduce_1, 
            self.conv_out_1, self.pool_1, self.reduce_num
        )
        
        net2 = self.compute_off_features(
            pre_off_list[1], self.conv_reduce_2, self.bn_reduce_2,
            self.conv_out_2, self.pool_2, self.reduce_num
        )
        
        net3 = self.compute_off_features(
            pre_off_list[2], self.conv_reduce_3, self.bn_reduce_3,
            self.conv_out_3, self.pool_3, self.reduce_num
        )
        
        # Cascade fusion
        net1 = F.interpolate(net1, size=net2.shape[-2:], mode='bilinear', align_corners=False)
        net_concat = torch.cat([net1, net2], dim=1)
        net_concat = self.conv_cascade_1(net_concat)
        net_concat = self.bn_cascade_1(net_concat)
        net_concat = self.relu(net_concat)
        
        net_concat = F.interpolate(net_concat, size=net3.shape[-2:], mode='bilinear', align_corners=False)
        net_concat = torch.cat([net_concat, net3], dim=1)
        net_concat = self.conv_cascade_2(net_concat)
        net_concat = self.bn_cascade_2(net_concat)
        net_concat = self.relu(net_concat)
        
        # Final layer
        net_out = self.conv_final(net_concat)
        net_out = self.bn_final(net_out)
        net_out = self.relu(net_out)
        
        return net_out


class SoftmaxNet(nn.Module):
    """Classification head for computing logits"""
    
    def __init__(self, input_features=1024, num_classes=2):
        super(SoftmaxNet, self).__init__()
        
        self.fc1 = nn.Linear(input_features, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) feature tensor
        Returns:
            logits: (B, num_classes) class logits
        """
        B = x.shape[0]
        x = x.view(B, -1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x


class LstmCnnNetMultiFrame(nn.Module):
    """Multi-frame LSTM+CNN Network for Face Anti-Spoofing"""
    
    def __init__(self, len_seq=5, num_classes=2, multiplier=2, single_ratio=0.5, input_features=1024):
        super(LstmCnnNetMultiFrame, self).__init__()
        
        self.len_seq = len_seq
        self.num_classes = num_classes
        self.single_ratio = single_ratio  # Weight for single-frame vs temporal
        self.alpha = 1 - single_ratio
        self.beta = single_ratio
        
        # Face map network for depth extraction
        self.face_map_net = FaceMapNet(multiplier=multiplier)
        
        # OFF network for temporal features
        self.off_net = OFFNet(len_seq, multiplier=multiplier, reduce_num=32)
        
        # Convolutional GRU for temporal processing
        self.conv_gru_net = ConvGRUNet(64 * multiplier, len_seq)
        
        # Classification head
        self.softmax_net = SoftmaxNet(input_features=input_features, num_classes=num_classes)
    
    def forward(self, x, is_training=True):
        """
        Args:
            x: (B, T, C, H, W) sequence of frames where T = len_seq
            is_training: bool, whether in training mode (checked via self.training)
        Returns:
            If training: (logits_cla, last_fused_map)
            If eval: logits_cla
        """
        B, T, C, H, W = x.shape
        
        # Reshape to batch format for FaceMapNet
        x_batch = x.view(B * T, C, H, W)
        
        # Extract depth maps and intermediate features
        features_map, pre_off_list = self.face_map_net(x_batch)
        
        # Get depth map spatial dimensions
        map_h, map_w = features_map.shape[-2:]
        
        # Reshape intermediate features to sequence format
        pre_off_list_seq = []
        for pre_off in pre_off_list:
            pre_off_seq = pre_off.view(B, T, pre_off.shape[1], pre_off.shape[2], pre_off.shape[3])
            pre_off_list_seq.append(pre_off_seq)
        
        # Compute OFF features
        off_features = self.off_net(pre_off_list)
        
        # Reshape OFF features to sequence format (B, T-1, C, H, W)
        off_features_seq = off_features.view(B, T - 1, off_features.shape[1], 
                                            off_features.shape[2], off_features.shape[3])
        
        # Process with ConvGRU (convert to time-first format)
        off_features_seq = off_features_seq.permute(1, 0, 2, 3, 4)  # (T-1, B, C, H, W)
        temporal_maps = self.conv_gru_net(off_features_seq)
        
        # Reshape single-frame depth maps
        single_maps = features_map.view(B, T, 1, map_h, map_w)
        
        # Fusion: Combine temporal and single-frame results
        final_maps = []
        for i in range(T - 1):
            fused = self.alpha * temporal_maps[i] + self.beta * single_maps[:, i, :, :, :]
            final_maps.append(fused)
        
        # Classification: Average all depth maps and pass through softmax net
        # Concatenate along channel axis: (B, T-1, H, W)
        logits_map_mean = torch.cat(final_maps, dim=1)
        # Average along channel axis: (B, 1, H, W)
        logits_map_mean = torch.mean(logits_map_mean, dim=1, keepdim=True)
        logits_cla = self.softmax_net(logits_map_mean)
        
        if self.training:
            return logits_cla, final_maps[-1]
        else:
            return logits_cla


# Example usage
if __name__ == "__main__":
    # Create model
    model = LstmCnnNetMultiFrame(len_seq=5, num_classes=2, single_ratio=0.5)
    
    # Create dummy input (batch_size=2, seq_len=5, channels=3, height=256, width=256)
    dummy_input = torch.randn(2, 5, 3, 256, 256)
    
    # Test evaluation mode
    model.eval()
    with torch.no_grad():
        eval_logits = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Eval output logits shape: {eval_logits.shape}")
    
    # Test training mode
    model.train()
    train_logits, train_map = model(dummy_input)
    print(f"Train output logits shape: {train_logits.shape}")
    print(f"Train output depth map shape: {train_map.shape}")
    print("Multi-frame model conversion and testing successful!")
