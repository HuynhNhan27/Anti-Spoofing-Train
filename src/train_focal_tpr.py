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
                # Positive class gets alpha, negative gets 1-alpha
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

def parse_args():
    parser = argparse.ArgumentParser(description="Anti-Spoofing Training with Focal Loss and TPR @ FPR=1%")
    parser.add_argument("--config", type=str, default="src/configs/resnet18_focal.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--data-dir", type=str, default=None, help="Override data directory path")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    return parser.parse_args()

def save_checkpoint(state, filepath):
    temp_filepath = filepath + ".tmp"
    torch.save(state, temp_filepath)
    os.replace(temp_filepath, filepath)

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_model(config, device):
    model_name = config["model"]["name"].lower()
    num_classes = config["model"]["num_classes"]
    pretrained = config["model"].get("pretrained", False)
    
    if model_name == "resnet18_fourier":
        from src.models.resnet_fourier import ResNet18Fourier
        model = ResNet18Fourier(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "resnet18":
        import torchvision.models as models
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(
            f"This training script only supports resnet18 and resnet18_fourier models. "
            f"Got: '{config['model']['name']}'"
        )
        
    # Handle multi-GPU if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
        
    model = model.to(device)
    return model

def compute_loss(model, outputs, labels, batch, device, criterions, use_fourier):
    """
    Computes loss dynamically.
    For Fourier model, combines Focal Classification loss and MSE Fourier auxiliary loss.
    """
    if use_fourier and isinstance(outputs, tuple):
        cls_output, ft_output = outputs
        ft_target = batch[1].to(device) # ft_sample is the second item in batch
        
        loss_cls = criterions["cls"](cls_output, labels)
        loss_ft = criterions["ft"](ft_output, ft_target)
        
        loss = 0.5 * loss_cls + 0.5 * loss_ft
        return loss, {"loss": loss.item(), "loss_cls": loss_cls.item(), "loss_ft": loss_ft.item()}
    else:
        cls_output = outputs[0] if isinstance(outputs, tuple) else outputs
        loss = criterions["cls"](cls_output, labels)
        return loss, {"loss": loss.item(), "loss_cls": loss.item()}

def calculate_tpr_at_fpr_1percent(labels, scores):
    """
    Calculates TPR @ FPR = 1% (BPCER = 1%) and APCER @ BPCER = 1%.
    Labels: 0 for Real/Live, 1 for Spoof/Fake.
    Scores: Probability of Spoof (class 1).
    """
    try:
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        
        # 1. Exact value by interpolation (standard ROC analysis)
        tpr_at_fpr_1pct_interp = float(np.interp(0.01, fpr, tpr))
        apcer_at_bpcer_1pct_interp = 1.0 - tpr_at_fpr_1pct_interp
        
        # 2. Conservative value where FPR <= 1% (ISO standard, thresholding based)
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
        print(f"Warning: ROC curve calculation failed: {e}. Using fallback values.")
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

def train_one_epoch(model, dataloader, optimizer, criterions, device, use_fourier):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    loss_cls_accum = 0.0
    loss_ft_accum = 0.0
    
    progress_bar = tqdm(dataloader, desc="  Train", leave=False)
    for batch in progress_bar:
        if use_fourier:
            inputs, _, labels = batch
        else:
            inputs, labels = batch
            
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss, loss_details = compute_loss(
            model, outputs, labels, batch, device, criterions, use_fourier
        )
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        loss_cls_accum += loss_details.get("loss_cls", 0.0) * inputs.size(0)
        loss_ft_accum += loss_details.get("loss_ft", 0.0) * inputs.size(0)
        
        cls_output = outputs[0] if isinstance(outputs, tuple) else outputs
        _, predicted = torch.max(cls_output.data, 1)
        
        all_labels.append(labels.detach())
        all_preds.append(predicted.detach())
        
        batch_acc = (predicted == labels).sum().item() / labels.size(0)
        progress_bar.set_postfix(loss=loss.item(), acc=100.0 * batch_acc)
        
    total = len(dataloader.dataset)
    epoch_loss = running_loss / total
    epoch_loss_cls = loss_cls_accum / total
    epoch_loss_ft = loss_ft_accum / total
    
    all_labels = torch.cat(all_labels).cpu().numpy()
    all_preds = torch.cat(all_preds).cpu().numpy()
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    
    return epoch_loss, epoch_loss_cls, epoch_loss_ft, epoch_acc, epoch_f1

def validate(model, dataloader, criterions, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_scores = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Val", leave=False):
            if len(batch) == 3:
                inputs, _, labels = batch
            else:
                inputs, labels = batch
                
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            outputs = model(inputs)
            loss = criterions["cls"](outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            
            # Predict labels (threshold = 0.5)
            _, predicted = torch.max(outputs.data, 1)
            
            # Compute probabilities for Spoof/Fake (class 1)
            probs = torch.softmax(outputs, dim=1)
            spoof_scores = probs[:, 1]
            
            all_labels.append(labels.detach())
            all_preds.append(predicted.detach())
            all_scores.append(spoof_scores.detach())
            
    val_loss = running_loss / len(dataloader.dataset)
    all_labels = torch.cat(all_labels).cpu().numpy()
    all_preds = torch.cat(all_preds).cpu().numpy()
    all_scores = torch.cat(all_scores).cpu().numpy()
    
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    apcer = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    bpcer = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    acer = (apcer + bpcer) / 2.0
    
    # Calculate TPR @ FPR = 1% (and APCER @ BPCER = 1%)
    tpr_metrics = calculate_tpr_at_fpr_1percent(all_labels, all_scores)
    
    metrics = {
        "loss": val_loss,
        "acc": val_acc,
        "f1": val_f1,
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": acer,
        "tpr_at_1pct_fpr_interp": tpr_metrics["tpr_at_1pct_fpr_interp"],
        "apcer_at_1pct_bpcer_interp": tpr_metrics["apcer_at_1pct_bpcer_interp"],
        "tpr_at_1pct_fpr_cons": tpr_metrics["tpr_at_1pct_fpr_cons"],
        "apcer_at_1pct_bpcer_cons": tpr_metrics["apcer_at_1pct_bpcer_cons"],
        "threshold_at_1pct_fpr": tpr_metrics["threshold_at_1pct_fpr"]
    }
    return metrics

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Set random seed for reproducibility
    seed = config["train"].get("seed", 42)
    set_seed(seed)
    
    # Override settings if command line args are passed
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["train"]["lr"] = args.lr
    if args.data_dir is not None:
        config["data"]["data_dir"] = args.data_dir
        
    device = torch.device(config["train"]["device"] if torch.cuda.is_available() and config["train"]["device"] == "cuda" else "cpu")
    print(f"Using device: {device}")
    
    is_kaggle_or_colab = (
        "KAGGLE_KERNEL_RUN_TYPE" in os.environ or
        "KAGGLE_URL_BASE" in os.environ or
        "google.colab" in sys.modules
    )
    
    num_workers = config["train"].get("num_workers", 4)
    if num_workers == "auto":
        if is_kaggle_or_colab:
            num_workers = 2
            print(f"Kaggle/Colab detected. Setting num_workers to {num_workers} to prevent CPU throttling.")
        else:
            num_workers = min(4, os.cpu_count() or 4)
            print(f"Local machine detected. Setting num_workers to: {num_workers}")
    else:
        num_workers = int(num_workers)
        
    print(f"Using {num_workers} data loader workers.")
    
    os.makedirs(config["train"]["save_dir"], exist_ok=True)
    os.makedirs(config["train"]["log_dir"], exist_ok=True)
    
    # 1. Initialize Dataloaders
    model_name_lower = config["model"]["name"].lower()
    use_fourier = config["data"]["use_fourier"] and "fourier" in model_name_lower
    
    train_loader = get_dataloader(
        data_dir=config["data"]["data_dir"],
        split="train",
        batch_size=config["train"]["batch_size"],
        input_size=config["data"]["input_size"],
        use_fourier=use_fourier,
        is_train=True,
        num_workers=num_workers
    )
    val_loader = get_dataloader(
        data_dir=config["data"]["data_dir"],
        split="val",
        batch_size=config["train"]["batch_size"],
        input_size=config["data"]["input_size"],
        use_fourier=use_fourier,
        is_train=False,
        num_workers=num_workers
    )
    
    # 2. Get Model
    model = get_model(config, device)
    
    # Compile model optionally (PyTorch 2.0+)
    torch_compile = config["train"].get("torch_compile", False)
    if torch_compile:
        try:
            print("Compiling model using torch.compile...")
            model = torch.compile(model)
            print("Model compiled successfully.")
        except Exception as e:
            print(f"Model compilation failed: {e}. Running uncompiled model.")
            
    # 3. Setup Criterions (Focal Loss for classification, MSE for Fourier)
    focal_gamma = config["train"].get("focal_loss_gamma", 2.0)
    focal_alpha = config["train"].get("focal_loss_alpha", None)
    print(f"Initializing Focal Loss (gamma={focal_gamma}, alpha={focal_alpha}) for classification.")
    
    criterions = {
        "cls": FocalLoss(alpha=focal_alpha, gamma=focal_gamma),
        "ft": nn.MSELoss()
    }
    
    # 4. Setup Optimizer and Scheduler
    model_params = model.module.parameters() if isinstance(model, nn.DataParallel) else model.parameters()
    optimizer = torch.optim.SGD(
        model_params,
        lr=config["train"]["lr"],
        momentum=config["train"]["momentum"],
        weight_decay=config["train"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config["train"]["milestones"],
        gamma=config["train"]["gamma"]
    )
    
    # 5. Initialize TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(config["train"]["log_dir"], config["model"]["name"]))
    
    # 6. Early Stopping & Checkpointing Configuration
    monitor_metric = config["train"].get("early_stopping_monitor", "tpr_at_1pct_fpr_interp")
    patience = config["train"].get("early_stopping_patience", 5)
    
    # Normalize early stopping metric name to allow user flexibility in config
    monitor_metric_norm = monitor_metric.lower().replace("@", "_at_").replace("=", "_").replace("%", "pct")
    if monitor_metric_norm in ["tpr_at_1pct_fpr", "tpr_at_fpr_1pct", "tpr_at_fpr_0.01", "tpr_at_1pct_fpr_interp"]:
        monitor_metric = "tpr_at_1pct_fpr_interp"
    elif monitor_metric_norm in ["tpr_at_1pct_fpr_cons", "tpr_at_fpr_1pct_cons", "tpr_at_fpr_0.01_cons"]:
        monitor_metric = "tpr_at_1pct_fpr_cons"
    elif monitor_metric_norm in ["apcer_at_1pct_bpcer", "apcer_at_bpcer_1pct", "apcer_at_bpcer_0.01", "apcer_at_1pct_bpcer_interp"]:
        monitor_metric = "apcer_at_1pct_bpcer_interp"
    elif monitor_metric_norm in ["apcer_at_1pct_bpcer_cons", "apcer_at_bpcer_1pct_cons", "apcer_at_bpcer_0.01_cons"]:
        monitor_metric = "apcer_at_1pct_bpcer_cons"
        
    higher_is_better = monitor_metric in [
        "f1", "acc", "accuracy", 
        "tpr_at_1pct_fpr_interp", "tpr_at_1pct_fpr_cons"
    ]
    best_val_metric = -1e9 if higher_is_better else 1e9
    epochs_no_improve = 0
    
    print(f"Monitoring metric '{monitor_metric}' (higher is better: {higher_is_better}) for checkpoint saving and early stopping (patience={patience}).")
    
    start_epoch = 0
    
    # Check if we should resume from a checkpoint
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming training from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith("module.") else k
                    new_state_dict[name] = v
                if isinstance(model, nn.DataParallel):
                    dp_state_dict = {}
                    for k, v in new_state_dict.items():
                        dp_state_dict[f"module.{k}"] = v
                    new_state_dict = dp_state_dict
                model.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(checkpoint)
                
            if isinstance(checkpoint, dict):
                if "optimizer" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                if "scheduler" in checkpoint:
                    scheduler.load_state_dict(checkpoint["scheduler"])
                if "epoch" in checkpoint:
                    start_epoch = checkpoint["epoch"]
                if "best_metric" in checkpoint:
                    best_val_metric = checkpoint["best_metric"]
            print(f"Resumed successfully. Starting training from epoch {start_epoch + 1}")
        else:
            print(f"Warning: Checkpoint not found at {args.resume}. Starting from scratch.")
            
    # 7. Training loop
    print(f"Starting training for model: {config['model']['name']} for {config['train']['epochs']} epochs...")
    
    try:
        for epoch in range(start_epoch, config["train"]["epochs"]):
            print(f"Epoch [{epoch+1}/{config['train']['epochs']}]")
            
            # Train
            train_loss, train_loss_cls, train_loss_ft, train_acc, train_f1 = train_one_epoch(
                model, train_loader, optimizer, criterions, device, use_fourier
            )
            
            # Validate
            val_metrics = validate(model, val_loader, criterions, device)
            
            scheduler.step()
            
            # Log to TensorBoard
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/train_cls", train_loss_cls, epoch)
            if use_fourier:
                writer.add_scalar("Loss/train_ft", train_loss_ft, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("F1_Score/train", train_f1, epoch)
            
            writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
            writer.add_scalar("Accuracy/val", val_metrics["acc"], epoch)
            writer.add_scalar("F1_Score/val", val_metrics["f1"], epoch)
            writer.add_scalar("APCER/val", val_metrics["apcer"], epoch)
            writer.add_scalar("BPCER/val", val_metrics["bpcer"], epoch)
            writer.add_scalar("ACER/val", val_metrics["acer"], epoch)
            
            writer.add_scalar("TPR_at_1pct_FPR_interp/val", val_metrics["tpr_at_1pct_fpr_interp"], epoch)
            writer.add_scalar("APCER_at_1pct_BPCER_interp/val", val_metrics["apcer_at_1pct_bpcer_interp"], epoch)
            writer.add_scalar("TPR_at_1pct_FPR_cons/val", val_metrics["tpr_at_1pct_fpr_cons"], epoch)
            writer.add_scalar("APCER_at_1pct_BPCER_cons/val", val_metrics["apcer_at_1pct_bpcer_cons"], epoch)
            writer.add_scalar("Threshold_at_1pct_FPR/val", val_metrics["threshold_at_1pct_fpr"], epoch)
            
            writer.add_scalar("Learning_Rate", optimizer.param_groups[0]["lr"], epoch)
            
            # Print epoch metrics
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Train F1: {train_f1*100:.2f}%")
            print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['acc'] * 100:.2f}% | Val F1: {val_metrics['f1'] * 100:.2f}%")
            print(f"  APCER: {val_metrics['apcer']*100:.2f}% | BPCER: {val_metrics['bpcer']*100:.2f}% | ACER: {val_metrics['acer']*100:.2f}%")
            print(f"  TPR @ FPR=1% (Interp): {val_metrics['tpr_at_1pct_fpr_interp']*100:.2f}% (APCER: {val_metrics['apcer_at_1pct_bpcer_interp']*100:.2f}%)")
            print(f"  TPR @ FPR=1% (Cons):   {val_metrics['tpr_at_1pct_fpr_cons']*100:.2f}% (APCER: {val_metrics['apcer_at_1pct_bpcer_cons']*100:.2f}%, Thresh: {val_metrics['threshold_at_1pct_fpr']:.4f})")
            
            current_metric_val = val_metrics.get(monitor_metric, val_metrics["tpr_at_1pct_fpr_interp"])
            
            # Save checkpoints
            state = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_metric": best_val_metric,
                "monitor_metric": monitor_metric
            }
            
            latest_path = os.path.join(config["train"]["save_dir"], f"{config['model']['name']}_latest.pth")
            save_checkpoint(state, latest_path)
            
            improved = False
            if higher_is_better:
                if current_metric_val > best_val_metric:
                    improved = True
            else:
                if current_metric_val < best_val_metric:
                    improved = True
                    
            if improved:
                best_val_metric = current_metric_val
                epochs_no_improve = 0
                best_path = os.path.join(config["train"]["save_dir"], f"{config['model']['name']}_best.pth")
                save_checkpoint(state, best_path)
                if monitor_metric in ["loss"]:
                    print(f"  ★ New best model saved with Val LOSS: {best_val_metric:.4f}")
                elif "apcer" in monitor_metric or "bpcer" in monitor_metric or "acer" in monitor_metric:
                    print(f"  ★ New best model saved with Val {monitor_metric.upper()}: {best_val_metric*100:.2f}%")
                else:
                    print(f"  ★ New best model saved with Val {monitor_metric.upper()}: {best_val_metric*100:.2f}%")
            else:
                epochs_no_improve += 1
                print(f"  Early stopping counter: {epochs_no_improve}/{patience}")
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered! Training stopped because validation {monitor_metric} did not improve for {patience} epochs.")
                    break
                    
        print("Training finished successfully!")
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user. Gracefully shutting down...")
        latest_path = os.path.join(config["train"]["save_dir"], f"{config['model']['name']}_latest.pth")
        best_path = os.path.join(config["train"]["save_dir"], f"{config['model']['name']}_best.pth")
        if os.path.exists(best_path):
            print(f"  ★ Best checkpoint up to the last completed epoch is saved at: {best_path}")
        if os.path.exists(latest_path):
            print(f"  ★ Latest checkpoint up to the last completed epoch is saved at: {latest_path}")
    finally:
        writer.close()
        print("Training session closed.")

if __name__ == "__main__":
    main()
