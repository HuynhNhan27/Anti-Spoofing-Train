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
from src.models.ssdg import SSDGModel


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


def train_one_epoch(
    model, 
    dataloader, 
    optimizer, 
    criterions, 
    device, 
    epoch, 
    total_epochs,
    alpha_adv, 
    beta_metric,
    dynamic_grl=True,
    max_lambda_grl=1.0
):
    model.train()
    running_loss = 0.0
    loss_cls_accum = 0.0
    loss_adv_accum = 0.0
    loss_compact_accum = 0.0
    loss_disperse_accum = 0.0
    
    all_labels = []
    all_preds = []
    
    total_steps = total_epochs * len(dataloader)
    progress_bar = tqdm(dataloader, desc="  Train SSDG", leave=False)
    
    for batch_idx, batch in enumerate(progress_bar):
        # 1. Unpack batch dictionary
        src_inputs = batch["source_img"].to(device, non_blocking=True)
        src_labels = batch["source_label"].to(device, non_blocking=True)
        tgt_inputs = batch["target_img"].to(device, non_blocking=True)
        tgt_labels = batch["target_label"].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # 2. Concatenate source and target batches
        images_all = torch.cat([src_inputs, tgt_inputs], dim=0)
        labels_all = torch.cat([src_labels, tgt_labels], dim=0)
        
        # Create domain labels: 0 for source, 1 for target
        domain_src = torch.zeros(src_inputs.size(0), dtype=torch.float, device=device)
        domain_tgt = torch.ones(tgt_inputs.size(0), dtype=torch.float, device=device)
        domain_all = torch.cat([domain_src, domain_tgt], dim=0)
        
        # Filter unlabeled target samples (label == -1)
        valid_mask = (labels_all != -1)
        if not valid_mask.any():
            continue
            
        images_all = images_all[valid_mask]
        labels_all = labels_all[valid_mask]
        domain_all = domain_all[valid_mask]
        
        # 3. Calculate dynamic GRL lambda_grl
        if dynamic_grl:
            current_step = epoch * len(dataloader) + batch_idx
            p = float(current_step) / total_steps
            # DANN GRL scheduling: scales from 0 to max_lambda_grl
            lambda_grl = (2.0 / (1.0 + np.exp(-10 * p)) - 1.0) * max_lambda_grl
        else:
            lambda_grl = max_lambda_grl
            
        # 4. Forward pass
        preds, f = model(images_all)
        
        # Loss 1: Classification Loss (Focal Loss) on all labeled samples
        loss_cls = criterions["cls"](preds, labels_all)
        
        # Create masks for Live/Real (0) and Spoof/Fake (1)
        live_mask = (labels_all == 0)
        spoof_mask = (labels_all == 1)
        
        f_live = f[live_mask]
        f_spoof = f[spoof_mask]
        domain_labels_live = domain_all[live_mask]
        
        # Loss 2: Single-Side Adversarial Loss (only for Live samples)
        loss_adv = torch.tensor(0.0, device=device)
        if f_live.size(0) > 0:
            domain_preds = model.D(f_live, alpha=lambda_grl)
            loss_adv = criterions["bce"](domain_preds.squeeze(-1), domain_labels_live)
            
        # Loss 3: Asymmetric Metric Loss
        loss_compact, loss_disperse = model.metric_loss(f_live, f_spoof)
        loss_metric = loss_compact + loss_disperse
        
        # Total loss aggregation
        total_loss = loss_cls + alpha_adv * loss_adv + beta_metric * loss_metric
        
        # 5. Backpropagation
        total_loss.backward()
        optimizer.step()
        
        # Metrics accumulation
        num_samples = images_all.size(0)
        running_loss += total_loss.item() * num_samples
        loss_cls_accum += loss_cls.item() * num_samples
        loss_adv_accum += loss_adv.item() * num_samples
        loss_compact_accum += loss_compact.item() * num_samples
        loss_disperse_accum += loss_disperse.item() * num_samples
        
        # Predictions accumulation (for classification accuracy)
        _, predicted = torch.max(preds.data, 1)
        all_labels.append(labels_all.detach())
        all_preds.append(predicted.detach())
        
        batch_acc = (predicted == labels_all).sum().item() / num_samples
        progress_bar.set_postfix(
            loss=total_loss.item(), 
            cls=loss_cls.item(), 
            adv=loss_adv.item(), 
            metric=loss_metric.item(),
            acc=100.0 * batch_acc
        )
        
    total = len(dataloader.dataset) * 2  # Each batch contains 1 source and 1 target batch
    # Clean up division in case some target images are unlabeled/filtered
    if len(all_labels) > 0:
        actual_total = sum(lbl.size(0) for lbl in all_labels)
        epoch_loss = running_loss / actual_total
        epoch_loss_cls = loss_cls_accum / actual_total
        epoch_loss_adv = loss_adv_accum / actual_total
        epoch_loss_compact = loss_compact_accum / actual_total
        epoch_loss_disperse = loss_disperse_accum / actual_total
        
        all_labels = torch.cat(all_labels).cpu().numpy()
        all_preds = torch.cat(all_preds).cpu().numpy()
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    else:
        epoch_loss, epoch_loss_cls, epoch_loss_adv, epoch_loss_compact, epoch_loss_disperse, epoch_acc, epoch_f1 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
    return (
        epoch_loss, 
        epoch_loss_cls, 
        epoch_loss_adv, 
        epoch_loss_compact, 
        epoch_loss_disperse, 
        epoch_acc, 
        epoch_f1
    )


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
            # Handle tuple returned by SSDGModel
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


def get_backbone_and_dim(config, device):
    model_name = config["model"]["name"].lower()
    pretrained = config["model"].get("pretrained", False)
    
    if model_name == "resnet18":
        import torchvision.models as models
        backbone = models.resnet18(pretrained=pretrained)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
    elif model_name in ["resnet34", "resnet50", "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "mobilenetv2", "mobilenet_v2"]:
        import timm
        timm_name = model_name
        if model_name == "mobilenetv2":
            timm_name = "mobilenetv2_100"
        backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=0)
        feat_dim = backbone.num_features
    else:
        raise ValueError(f"Unsupported backbone for SSDG: {model_name}")
        
    backbone = backbone.to(device)
    return backbone, feat_dim


def main():
    parser = argparse.ArgumentParser(description="Single-Side Domain Generalization (SSDG) for Face Anti-Spoofing")
    parser.add_argument("--config", type=str, default="src/configs/resnet18_ssdg.yaml", help="Path to config YAML")
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
    
    # SSDG settings from config
    alpha_adv = config["train"].get("alpha", 0.1)
    beta_metric = config["train"].get("beta", 0.5)
    margin = config["train"].get("margin", 1.0)
    dist_type = config["train"].get("dist_type", "mse")
    lambda_grl = config["train"].get("lambda_grl", 1.0)
    dynamic_grl = config["train"].get("dynamic_grl", True)
    
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
    
    # 2. Get Model
    backbone, feat_dim = get_backbone_and_dim(config, device)
    model = SSDGModel(
        backbone=backbone,
        feat_dim=feat_dim,
        feature_dim=512,
        margin=margin,
        dist_type=dist_type
    )
    
    # Unwrap DataParallel if present
    if isinstance(model, nn.DataParallel):
        print("SSDG training: Unwrapping DataParallel for correct feature-based alignment.")
        model = model.module
        
    model = model.to(device)
    
    # 3. Setup Criterions (Focal Loss for classification, BCE for Domain, Metric loss is internal to model)
    focal_gamma = config["train"].get("focal_loss_gamma", 2.0)
    focal_alpha = config["train"].get("focal_loss_alpha", None)
    
    print(f"Initializing Focal Loss (gamma={focal_gamma}, alpha={focal_alpha}) for classification.")
    criterions = {
        "cls": FocalLoss(alpha=focal_alpha, gamma=focal_gamma),
        "bce": nn.BCEWithLogitsLoss()
    }
    
    # 4. Setup Optimizer & Scheduler (incorporating C_real automatically from model.parameters())
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
    
    writer = SummaryWriter(log_dir=os.path.join(config["train"]["log_dir"], f"{config['model']['name']}_ssdg"))
    
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
        
    print(f"Starting SSDG training for {config['train']['epochs']} epochs...")
    try:
        for epoch in range(start_epoch, config["train"]["epochs"]):
            print(f"Epoch [{epoch+1}/{config['train']['epochs']}]")
            
            # Train
            train_loss, train_loss_cls, train_loss_adv, train_loss_compact, train_loss_disperse, train_acc, train_f1 = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                criterions=criterions,
                device=device,
                epoch=epoch,
                total_epochs=config["train"]["epochs"],
                alpha_adv=alpha_adv,
                beta_metric=beta_metric,
                dynamic_grl=dynamic_grl,
                max_lambda_grl=lambda_grl
            )
            
            # Validate Source
            val_src = validate(model, val_loader_source, criterions["cls"], device, domain_name="source")
            # Validate Target
            val_tgt = validate(model, val_loader_target, criterions["cls"], device, domain_name="target")
            
            scheduler.step()
            
            # TensorBoard logging
            writer.add_scalar("Loss/train_total", train_loss, epoch)
            writer.add_scalar("Loss/train_cls", train_loss_cls, epoch)
            writer.add_scalar("Loss/train_adv", train_loss_adv, epoch)
            writer.add_scalar("Loss/train_compact", train_loss_compact, epoch)
            writer.add_scalar("Loss/train_disperse", train_loss_disperse, epoch)
            writer.add_scalar("Accuracy/train_all", train_acc, epoch)
            
            for key, val in val_src.items():
                writer.add_scalar(f"Source_Val/{key}", val, epoch)
            for key, val in val_tgt.items():
                writer.add_scalar(f"Target_Val/{key}", val, epoch)
                
            print(f"  Train Total Loss: {train_loss:.4f} | Cls: {train_loss_cls:.4f} | Adv: {train_loss_adv:.4f} | Metric (Comp/Disp): {train_loss_compact:.4f}/{train_loss_disperse:.4f} | Acc: {train_acc*100:.2f}%")
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
            
            latest_path = os.path.join(config["train"]["save_dir"], f"{config['model']['name']}_ssdg_latest.pth")
            torch.save(state, latest_path)
            
            if current_metric > best_val_metric:
                best_val_metric = current_metric
                epochs_no_improve = 0
                best_path = os.path.join(config["train"]["save_dir"], f"{config['model']['name']}_ssdg_best.pth")
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
        writer.close()


if __name__ == "__main__":
    main()
