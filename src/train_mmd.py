import os
import random
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_curve

# Add project root to python path to resolve src.* imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import get_dataloader
from src.data.domain_dataset import get_domain_balanced_dataloader, get_target_dataloader

class FocalLoss(nn.Module):
    """
    Focal Loss for binary/multi-class classification.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # pt is probability of correct class
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
                focal_loss = alpha_t * focal_loss
            elif isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.to(inputs.device)[targets]
                focal_loss = alpha_t * focal_loss
            elif isinstance(self.alpha, (list, tuple)):
                alpha_t = torch.tensor(self.alpha, device=inputs.device)[targets]
                focal_loss = alpha_t * focal_loss
                
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MMDLoss(nn.Module):
    """
    Maximum Mean Discrepancy (MMD) Loss with multi-kernel RBF support.
    Supports global standard MMD and class-wise Conditional MMD (CMMD).
    
    This implementation estimates a single joint kernel bandwidth (sigma) from
    the combined source and target samples, ensuring that within-domain (xx, yy)
    and cross-domain (xy) kernels are mapped into the exact same RKHS.
    """
    def __init__(self, kernels=[1.0, 5.0, 10.0], align_mode="conditional"):
        super(MMDLoss, self).__init__()
        self.kernels = kernels
        self.align_mode = align_mode

    def get_pairwise_distances(self, x, y):
        # x: [N, D], y: [M, D]
        x_sq = torch.sum(x ** 2, dim=1, keepdim=True)
        y_sq = torch.sum(y ** 2, dim=1, keepdim=True).t()
        dist = x_sq + y_sq - 2.0 * torch.matmul(x, y.t())
        # Clamping to avoid tiny negative values due to numerical precision
        return torch.clamp(dist, min=0.0)

    def mmd_distance(self, x, y):
        # Compute joint pairwise distances
        dist_xx = self.get_pairwise_distances(x, x)
        dist_yy = self.get_pairwise_distances(y, y)
        dist_xy = self.get_pairwise_distances(x, y)
        
        # Estimate a single joint base_sigma from the combined set of x and y
        # This keeps the kernel bandwidth mathematically consistent across all terms
        with torch.no_grad():
            combined = torch.cat([x, y], dim=0)
            dist_combined = self.get_pairwise_distances(combined, combined)
            base_sigma = torch.mean(dist_combined)
            if base_sigma == 0:
                base_sigma = 1.0
                
        xx = 0.0
        yy = 0.0
        xy = 0.0
        for k in self.kernels:
            bandwidth = base_sigma * k
            xx += torch.exp(-dist_xx / (2.0 * bandwidth))
            yy += torch.exp(-dist_yy / (2.0 * bandwidth))
            xy += torch.exp(-dist_xy / (2.0 * bandwidth))
            
        return xx.mean() + yy.mean() - 2.0 * xy.mean()

    def forward(self, source_features, source_labels, target_features, target_labels):
        if self.align_mode == "conditional":
            # Conditional MMD (class-wise alignment)
            # Find classes available in the batch (excluding target unlabeled -1)
            valid_tgt_mask = (target_labels != -1)
            if not valid_tgt_mask.any():
                return torch.tensor(0.0, device=source_features.device)
                
            classes = torch.unique(source_labels)
            loss = 0.0
            count = 0
            
            for c in classes:
                src_mask = (source_labels == c)
                tgt_mask = (target_labels == c) & valid_tgt_mask
                
                if src_mask.sum() > 0 and tgt_mask.sum() > 0:
                    src_feat_c = source_features[src_mask]
                    tgt_feat_c = target_features[tgt_mask]
                    loss += self.mmd_distance(src_feat_c, tgt_feat_c)
                    count += 1
                    
            if count > 0:
                return loss / count
            return torch.tensor(0.0, device=source_features.device)
        else:
            # Global standard MMD
            return self.mmd_distance(source_features, target_features)


class FeatureExtractorHook:
    """
    Model-agnostic forward pre-hook on the classification layer to extract intermediate features.
    """
    def __init__(self, model):
        self.model = model
        self.features = []
        self.hook_handle = None
        self._register_hook()

    def _register_hook(self):
        clf_layer = None
        
        # 1. Search by known properties
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'logits'):
            clf_layer = self.model.model.logits
        elif hasattr(self.model, 'logits'):
            clf_layer = self.model.logits
        elif hasattr(self.model, 'fc'):
            clf_layer = self.model.fc
        elif hasattr(self.model, 'classifier'):
            clf_layer = self.model.classifier
            if isinstance(clf_layer, nn.Sequential):
                for module in clf_layer:
                    if isinstance(module, nn.Linear):
                        clf_layer = module
                        break
                        
        # 2. Fallback search (find last Linear layer)
        if clf_layer is None:
            linears = []
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    linears.append(module)
            if linears:
                clf_layer = linears[-1]
                
        if clf_layer is not None:
            print(f"FeatureExtractorHook: Hooked layer: {clf_layer}")
            self.hook_handle = clf_layer.register_forward_pre_hook(self._hook_fn)
        else:
            raise ValueError("FeatureExtractorHook: Could not identify classification layer in model.")

    def _hook_fn(self, module, input):
        # Only extract features during training to prevent memory leaks in validation
        if not self.model.training:
            return
        feat = input[0]
        if len(feat.shape) > 2:
            feat = feat.view(feat.size(0), -1)
        self.features.append(feat)

    def clear(self):
        self.features.clear()

    def remove(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()


def calculate_tpr_at_fpr_1percent(labels, scores):
    """
    Calculates TPR @ FPR = 1% (BPCER = 1%) and APCER @ BPCER = 1%.
    Labels: 0 for Real/Live, 1 for Spoof/Fake.
    Scores: Probability of Spoof (class 1).
    """
    try:
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        tpr_at_fpr_1pct_interp = float(np.interp(0.01, fpr, tpr))
        apcer_at_bpcer_1pct_interp = 1.0 - tpr_at_fpr_1pct_interp
        
        indices = np.where(fpr <= 0.01)[0]
        if len(indices) > 0:
            idx = indices[-1]
            tpr_at_fpr_1pct_cons = float(tpr[idx])
            apcer_at_bpcer_1pct_cons = 1.0 - tpr_at_fpr_1pct_cons
            threshold_at_fpr_1pct = float(thresholds[idx])
        else:
            tpr_at_fpr_1pct_cons = 0.0
            apcer_at_bpcer_1pct_cons = 1.0
            threshold_at_fpr_1pct = 1.0
    except Exception as e:
        tpr_at_fpr_1pct_interp = 0.0
        apcer_at_bpcer_1pct_interp = 1.0
        tpr_at_fpr_1pct_cons = 0.0
        apcer_at_bpcer_1pct_cons = 1.0
        threshold_at_fpr_1pct = 0.5
        
    return {
        "tpr_at_1pct_fpr_interp": tpr_at_fpr_1pct_interp,
        "apcer_at_1pct_bpcer_interp": apcer_at_bpcer_1pct_interp,
        "tpr_at_1pct_fpr_cons": tpr_at_fpr_1pct_cons,
        "apcer_at_1pct_bpcer_cons": apcer_at_bpcer_1pct_cons,
        "threshold_at_1pct_fpr": threshold_at_fpr_1pct
    }


def compute_cls_loss(outputs, labels, batch, device, criterions, use_fourier, is_source=True):
    """
    Computes classification loss (+ Fourier MSE loss if fourier auxiliary is enabled).
    """
    if use_fourier and isinstance(outputs, tuple):
        cls_output, ft_output = outputs
        ft_key = "source_ft" if is_source else "target_ft"
        ft_target = batch[ft_key].to(device)
        
        loss_cls = criterions["cls"](cls_output, labels)
        loss_ft = criterions["ft"](ft_output, ft_target)
        
        loss = 0.85 * loss_cls + 0.15 * loss_ft
        return loss, {"loss_cls": loss_cls.item(), "loss_ft": loss_ft.item()}
    else:
        cls_output = outputs[0] if isinstance(outputs, tuple) else outputs
        loss = criterions["cls"](cls_output, labels)
        return loss, {"loss_cls": loss.item()}


def train_one_epoch(model, dataloader, optimizer, criterions, device, feature_extractor, use_fourier, lambda_mmd, target_cls_weight):
    model.train()
    running_loss = 0.0
    loss_cls_src_accum = 0.0
    loss_cls_tgt_accum = 0.0
    loss_ft_src_accum = 0.0
    loss_mmd_accum = 0.0
    
    all_src_labels = []
    all_src_preds = []
    
    progress_bar = tqdm(dataloader, desc="  Train Domain Adaptation", leave=False)
    for batch in progress_bar:
        # Unpack batch dictionary
        src_inputs = batch["source_img"].to(device, non_blocking=True)
        src_labels = batch["source_label"].to(device, non_blocking=True)
        tgt_inputs = batch["target_img"].to(device, non_blocking=True)
        tgt_labels = batch["target_label"].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        feature_extractor.clear()
        
        # 1. Forward source batch
        src_outputs = model(src_inputs)
        # 2. Forward target batch
        tgt_outputs = model(tgt_inputs)
        
        # Extract features recorded by hook
        src_features = feature_extractor.features[0]
        tgt_features = feature_extractor.features[1]
        
        # 3. Compute classification and Fourier loss on Source
        loss_cls_src, src_details = compute_cls_loss(
            src_outputs, src_labels, batch, device, criterions, use_fourier, is_source=True
        )
        
        # 4. Compute classification loss on Target (if labeled and enabled)
        loss_cls_tgt = torch.tensor(0.0, device=device)
        tgt_details = {}
        if target_cls_weight > 0.0:
            valid_tgt_mask = (tgt_labels != -1)
            if valid_tgt_mask.any():
                valid_tgt_outputs = tgt_outputs[0][valid_tgt_mask] if isinstance(tgt_outputs, tuple) else tgt_outputs[valid_tgt_mask]
                valid_tgt_labels = tgt_labels[valid_tgt_mask]
                loss_cls_tgt, tgt_details = compute_cls_loss(
                    valid_tgt_outputs, valid_tgt_labels, batch, device, criterions, use_fourier=False, is_source=False
                )
                
        # 5. Compute MMD / CMMD loss
        loss_mmd = criterions["mmd"](src_features, src_labels, tgt_features, tgt_labels)
        
        # Total loss calculation
        total_loss = loss_cls_src + target_cls_weight * loss_cls_tgt + lambda_mmd * loss_mmd
        
        total_loss.backward()
        optimizer.step()
        
        # Clear extractor features immediately to free up GPU memory references
        feature_extractor.clear()
        
        # Metrics accumulation
        running_loss += total_loss.item() * src_inputs.size(0)
        loss_cls_src_accum += src_details.get("loss_cls", 0.0) * src_inputs.size(0)
        loss_cls_tgt_accum += tgt_details.get("loss_cls", 0.0) * src_inputs.size(0)
        loss_ft_src_accum += src_details.get("loss_ft", 0.0) * src_inputs.size(0)
        loss_mmd_accum += loss_mmd.item() * src_inputs.size(0)
        
        # Source Accuracy
        src_cls_output = src_outputs[0] if isinstance(src_outputs, tuple) else src_outputs
        _, predicted = torch.max(src_cls_output.data, 1)
        all_src_labels.append(src_labels.detach())
        all_src_preds.append(predicted.detach())
        
        batch_acc = (predicted == src_labels).sum().item() / src_labels.size(0)
        progress_bar.set_postfix(loss=total_loss.item(), mmd=loss_mmd.item(), acc=100.0 * batch_acc)
        
    total = len(dataloader.dataset)
    epoch_loss = running_loss / total
    epoch_loss_cls_src = loss_cls_src_accum / total
    epoch_loss_cls_tgt = loss_cls_tgt_accum / total
    epoch_loss_ft_src = loss_ft_src_accum / total
    epoch_loss_mmd = loss_mmd_accum / total
    
    all_src_labels = torch.cat(all_src_labels).cpu().numpy()
    all_src_preds = torch.cat(all_src_preds).cpu().numpy()
    epoch_acc_src = accuracy_score(all_src_labels, all_src_preds)
    epoch_f1_src = f1_score(all_src_labels, all_src_preds, average='binary', zero_division=0)
    
    return epoch_loss, epoch_loss_cls_src, epoch_loss_cls_tgt, epoch_loss_ft_src, epoch_loss_mmd, epoch_acc_src, epoch_f1_src


def validate(model, dataloader, criterion, device, domain_name="val"):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_scores = []
    all_preds = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"  Val ({domain_name})", leave=False):
            # Support dict batches or standard loader tuples
            if isinstance(batch, dict):
                inputs = batch["target_img"] if domain_name == "target" else batch["source_img"]
                labels = batch["target_label"] if domain_name == "target" else batch["source_label"]
            elif len(batch) == 3:
                inputs, _, labels = batch
            else:
                inputs, labels = batch
                
            # Filter unlabeled target samples
            valid_mask = (labels != -1)
            if not valid_mask.any():
                continue
                
            inputs = inputs[valid_mask].to(device, non_blocking=True)
            labels = labels[valid_mask].to(device, non_blocking=True)
            
            outputs = model(inputs)
            cls_output = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = criterion(cls_output, labels)
            
            running_loss += loss.item() * inputs.size(0)
            
            probs = F.softmax(cls_output, dim=1)
            scores = probs[:, 1]
            _, predicted = torch.max(cls_output.data, 1)
            
            all_labels.append(labels.detach())
            all_scores.append(scores.detach())
            all_preds.append(predicted.detach())
            
    if not all_labels:
        return {
            "loss": 0.0, "acc": 0.0, "f1": 0.0, "apcer": 0.0, "bpcer": 0.0, "acer": 0.0,
            "tpr_at_1pct_fpr_interp": 0.0, "apcer_at_1pct_bpcer_interp": 1.0,
            "tpr_at_1pct_fpr_cons": 0.0, "apcer_at_1pct_bpcer_cons": 1.0
        }
        
    all_labels = torch.cat(all_labels).cpu().numpy()
    all_scores = torch.cat(all_scores).cpu().numpy()
    all_preds = torch.cat(all_preds).cpu().numpy()
    
    val_loss = running_loss / len(all_labels)
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    apcer = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    bpcer = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    acer = (apcer + bpcer) / 2.0
    
    tpr_metrics = calculate_tpr_at_fpr_1percent(all_labels, all_scores)
    
    return {
        "loss": val_loss,
        "acc": val_acc,
        "f1": val_f1,
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": acer,
        **tpr_metrics
    }


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(config, device):
    model_name = config["model"]["name"].lower()
    num_classes = config["model"]["num_classes"]
    pretrained = config["model"].get("pretrained", False)
    
    if model_name == "resnet18":
        import torchvision.models as models
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(device)
    else:
        # Fallback to importing get_model from original train script
        from src.train import get_model as get_original_model
        model = get_original_model(config, device)
        
    # If the model was wrapped in nn.DataParallel, unwrap it.
    # Hooking classification layers across multiple GPUs in DataParallel threads can result in
    # race conditions and out-of-order feature gathering, which breaks class-wise CMMD alignment.
    if isinstance(model, nn.DataParallel):
        print("FeatureExtractorHook: Unwrapping DataParallel model for correct feature alignment.")
        model = model.module
        
    return model


def main():
    parser = argparse.ArgumentParser(description="Anti-Spoofing Domain Adaptation with Conditional MMD")
    parser.add_argument("--config", type=str, default="src/configs/resnet18_mmd.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    args = parser.parse_args()
    
    config = yaml.safe_load(open(args.config, "r"))
    
    set_seed(config["train"].get("seed", 42))
    
    if args.epochs:
        config["train"]["epochs"] = args.epochs
    if args.batch_size:
        config["train"]["batch_size"] = args.batch_size
    if args.lr:
        config["train"]["lr"] = args.lr
        
    device = torch.device(config["train"]["device"] if torch.cuda.is_available() and config["train"]["device"] == "cuda" else "cpu")
    print(f"Using device: {device}")
    
    num_workers = config["train"].get("num_workers", 4)
    if num_workers == "auto":
        num_workers = min(4, os.cpu_count() or 4)
    else:
        num_workers = int(num_workers)
        
    os.makedirs(config["train"]["save_dir"], exist_ok=True)
    os.makedirs(config["train"]["log_dir"], exist_ok=True)
    
    # 1. Initialize Balanced DataLoader
    use_fourier = config["data"]["use_fourier"] and (config["model"]["name"].lower() == "minifasv2" or "fourier" in config["model"]["name"].lower())
    use_randaugment = config["data"].get("use_randaugment", False)
    ra_num_ops = config["data"].get("ra_num_ops", 2)
    ra_magnitude = config["data"].get("ra_magnitude", 9)
    
    train_loader = get_domain_balanced_dataloader(
        source_dir=config["data"]["data_dir"],
        target_dir=config["data"]["target_dir"],
        split="train",
        batch_size=config["train"]["batch_size"],
        input_size=config["data"]["input_size"],
        target_csv_path=config["data"].get("target_train_csv"),
        use_fourier=use_fourier,
        is_train=True,
        num_workers=num_workers,
        use_randaugment=use_randaugment,
        ra_num_ops=ra_num_ops,
        ra_magnitude=ra_magnitude
    )
    
    # Validation loaders
    val_loader_source = get_dataloader(
        data_dir=config["data"]["data_dir"],
        split="val",
        batch_size=config["train"]["batch_size"],
        input_size=config["data"]["input_size"],
        use_fourier=use_fourier,
        is_train=False,
        num_workers=num_workers
    )
    
    # Target validation loader (using CSV if specified)
    target_val_csv = config["data"].get("target_val_csv")
    val_loader_target = get_target_dataloader(
        target_dir=config["data"]["target_dir"],
        split="val",
        batch_size=config["train"]["batch_size"],
        input_size=config["data"]["input_size"],
        csv_path=target_val_csv,
        use_fourier=use_fourier,
        is_train=False,
        num_workers=num_workers
    )
    
    # 2. Get Model and Hook
    model = get_model(config, device)
    feature_extractor = FeatureExtractorHook(model)
    
    # 3. Setup Criterions (Focal Loss for classification, MSE for Fourier, MMD Loss)
    focal_gamma = config["train"].get("focal_loss_gamma", 2.0)
    focal_alpha = config["train"].get("focal_loss_alpha", None)
    
    print(f"Initializing Focal Loss (gamma={focal_gamma}, alpha={focal_alpha}) for classification.")
    criterions = {
        "cls": FocalLoss(alpha=focal_alpha, gamma=focal_gamma),
        "ft": nn.MSELoss(),
        "mmd": MMDLoss(align_mode=config["train"].get("align_mode", "conditional"))
    }
    
    # 4. Setup Optimizer & Scheduler
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["train"]["lr"],
        momentum=config["train"]["momentum"],
        weight_decay=config["train"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config["train"]["milestones"],
        gamma=config["train"]["gamma"]
    )
    
    writer = SummaryWriter(log_dir=os.path.join(config["train"]["log_dir"], f"{config['model']['name']}_mmd"))
    
    # Early stopping config
    monitor_metric = config["train"].get("early_stopping_monitor", "tpr_at_1pct_fpr_interp")
    patience = config["train"].get("early_stopping_patience", 5)
    best_val_metric = -1e9
    epochs_no_improve = 0
    
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming training from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]
        best_val_metric = checkpoint.get("best_metric", -1e9)
        print(f"Resumed successfully at epoch {start_epoch + 1}")
        
    print(f"Starting training for {config['train']['epochs']} epochs...")
    try:
        for epoch in range(start_epoch, config["train"]["epochs"]):
            print(f"Epoch [{epoch+1}/{config['train']['epochs']}]")
            
            # Train
            train_loss, train_loss_cls_src, train_loss_cls_tgt, train_loss_ft_src, train_loss_mmd, train_acc_src, train_f1_src = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                criterions=criterions,
                device=device,
                feature_extractor=feature_extractor,
                use_fourier=use_fourier,
                lambda_mmd=config["train"].get("lambda_mmd", 0.1),
                target_cls_weight=config["train"].get("target_cls_weight", 0.5)
            )
            
            # Validate Source
            val_src = validate(model, val_loader_source, criterions["cls"], device, domain_name="source")
            # Validate Target
            val_tgt = validate(model, val_loader_target, criterions["cls"], device, domain_name="target")
            
            scheduler.step()
            
            # TensorBoard logging
            writer.add_scalar("Loss/train_total", train_loss, epoch)
            writer.add_scalar("Loss/train_cls_src", train_loss_cls_src, epoch)
            writer.add_scalar("Loss/train_cls_tgt", train_loss_cls_tgt, epoch)
            writer.add_scalar("Loss/train_mmd", train_loss_mmd, epoch)
            writer.add_scalar("Accuracy/train_src", train_acc_src, epoch)
            
            for key, val in val_src.items():
                writer.add_scalar(f"Source_Val/{key}", val, epoch)
            for key, val in val_tgt.items():
                writer.add_scalar(f"Target_Val/{key}", val, epoch)
                
            print(f"  Train Total Loss: {train_loss:.4f} | MMD: {train_loss_mmd:.4f} | Src Acc: {train_acc_src*100:.2f}%")
            print(f"  Source Val Loss: {val_src['loss']:.4f} | Src Val Acc: {val_src['acc']*100:.2f}% | Src Val ACER: {val_src['acer']*100:.2f}%")
            print(f"  Target Val Loss: {val_tgt['loss']:.4f} | Tgt Val Acc: {val_tgt['acc']*100:.2f}% | Tgt Val ACER: {val_tgt['acer']*100:.2f}% | TPR@FPR=1%: {val_tgt['tpr_at_1pct_fpr_interp']*100:.2f}%")
            
            # Check improvement based on target validation metric (or source)
            current_metric = val_tgt.get(monitor_metric, val_tgt["tpr_at_1pct_fpr_interp"])
            state = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_metric": best_val_metric
            }
            
            latest_path = os.path.join(config["train"]["save_dir"], f"{config['model']['name']}_mmd_latest.pth")
            torch.save(state, latest_path)
            
            if current_metric > best_val_metric:
                best_val_metric = current_metric
                epochs_no_improve = 0
                best_path = os.path.join(config["train"]["save_dir"], f"{config['model']['name']}_mmd_best.pth")
                torch.save(state, best_path)
                print(f"  ★ New best model saved on target validation {monitor_metric}: {best_val_metric*100:.2f}%")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered. No improvement on target validation '{monitor_metric}' for {patience} epochs.")
                    break
                    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Shutting down gracefully...")
    finally:
        feature_extractor.remove()
        writer.close()

if __name__ == "__main__":
    main()
