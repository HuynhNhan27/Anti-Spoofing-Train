import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Gradient reversal: reverse gradient and scale by alpha
        return grad_output.neg() * ctx.alpha, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversal.apply(x, self.alpha)


class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256):
        super(DomainDiscriminator, self).__init__()
        self.grl = GradientReversalLayer()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x, alpha=1.0):
        self.grl.alpha = alpha
        x_rev = self.grl(x)
        return self.net(x_rev)


class AsymmetricMetricLoss(nn.Module):
    def __init__(self, feature_dim=512, margin=1.0, dist_type="mse"):
        super(AsymmetricMetricLoss, self).__init__()
        self.margin = margin
        self.dist_type = dist_type
        # Learnable center C_real representing Live faces
        self.c_real = nn.Parameter(torch.randn(1, feature_dim))
        
    def forward(self, f_live, f_spoof):
        loss_compact = torch.tensor(0.0, device=self.c_real.device)
        loss_disperse = torch.tensor(0.0, device=self.c_real.device)
        
        if f_live.size(0) > 0:
            if self.dist_type == "mse":
                # Mean squared error: mean along dimension of squared difference
                dist_live = torch.mean((f_live - self.c_real) ** 2, dim=1)
            elif self.dist_type == "cosine":
                dist_live = 1.0 - F.cosine_similarity(f_live, self.c_real, dim=1)
            elif self.dist_type == "euclidean":
                dist_live = torch.norm(f_live - self.c_real, p=2, dim=1)
            else:
                raise ValueError(f"Unknown dist_type: {self.dist_type}")
            loss_compact = dist_live.mean()
            
        if f_spoof.size(0) > 0:
            if self.dist_type == "mse":
                dist_spoof = torch.mean((f_spoof - self.c_real) ** 2, dim=1)
            elif self.dist_type == "cosine":
                dist_spoof = 1.0 - F.cosine_similarity(f_spoof, self.c_real, dim=1)
            elif self.dist_type == "euclidean":
                dist_spoof = torch.norm(f_spoof - self.c_real, p=2, dim=1)
            else:
                raise ValueError(f"Unknown dist_type: {self.dist_type}")
            
            # Dispersion pushes spoof features away from C_real to be >= margin
            loss_disperse = torch.clamp(self.margin - dist_spoof, min=0.0).mean()
            
        return loss_compact, loss_disperse


class SSDGModel(nn.Module):
    def __init__(self, backbone, feat_dim, feature_dim=512, margin=1.0, dist_type="mse"):
        super(SSDGModel, self).__init__()
        self.backbone = backbone
        self.bottleneck = nn.Sequential(
            nn.Linear(feat_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.PReLU()
        )
        self.C = nn.Linear(feature_dim, 2)
        self.D = DomainDiscriminator(input_dim=feature_dim)
        self.metric_loss = AsymmetricMetricLoss(feature_dim=feature_dim, margin=margin, dist_type=dist_type)
        
    def forward(self, x, alpha=1.0):
        # Extract backbone features
        feats = self.backbone(x)
        # Flatten if necessary
        if len(feats.shape) > 2:
            feats = feats.view(feats.size(0), -1)
        f = self.bottleneck(feats)
        preds = self.C(f)
        return preds, f
