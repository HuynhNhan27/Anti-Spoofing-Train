import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOHead(nn.Module):
    """
    Projection and Prototype Head for DINO Self-Distillation.
    Standard MLP design with a bottleneck followed by L2 normalization and a weight-normalized output layer.
    """
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, bottleneck_dim=256):
        super().__init__()
        # 3-layer MLP
        hidden_dim = 2048
        
        layers = []
        # Layer 1
        layers.append(nn.Linear(in_dim, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        # Layer 2
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        # Layer 3 (bottleneck)
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Prototype layer mapping bottleneck to out_dim (classes/prototypes)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        # 1. Forward through MLP
        x = self.mlp(x)
        # 2. Normalize features along representation dimension
        x = F.normalize(x, dim=-1, p=2)
        # 3. Project to prototype space
        x = self.last_layer(x)
        return x

class MultiCropWrapper(nn.Module):
    """
    Wrapper mapping multiple crop tensors to the backbone and projection head.
    Optimized by grouping views of the same resolution into a single forward pass.
    """
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # If input is a single tensor (e.g. during typical inference or eval)
        if not isinstance(x, list):
            return self.head(self.backbone(x))

        # Separate crops into global and local crops.
        # First 2 crops are global (224x224), remaining crops are local (96x96).
        n_global = 2
        
        # Concat global crops and run forward
        global_crops = torch.cat(x[:n_global], dim=0) # shape: (2 * B, C, 224, 224)
        global_feats = self.backbone(global_crops)
        global_out = self.head(global_feats)
        
        # Concat local crops and run forward
        if len(x) > n_global:
            local_crops = torch.cat(x[n_global:], dim=0) # shape: ((num_crops - 2) * B, C, 96, 96)
            local_feats = self.backbone(local_crops)
            local_out = self.head(local_feats)
            # Combine global and local outputs
            return torch.cat([global_out, local_out], dim=0)
        
        return global_out
