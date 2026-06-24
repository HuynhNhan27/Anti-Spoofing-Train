import os
import sys
# Add project root to python path to resolve src.* imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from src.data.dataset import get_dataloader

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Anti-Spoofing Model Training")
    parser.add_argument("--config", type=str, default="src/configs/minifasv2.yaml", help="Path to config YAML")
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
    
    if model_name == "minifasv2":
        from src.models.minifasv2.model import MultiFTNet
        input_size = config["data"]["input_size"]
        k_size = (input_size + 15) // 16
        conv6_kernel = (k_size, k_size)
        model = MultiFTNet(
            num_channels=3,
            num_classes=num_classes,
            embedding_size=config["model"]["embedding_size"],
            conv6_kernel=conv6_kernel
        )
    elif model_name == "resnet18":
        import torchvision.models as models
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name in ["resnet34", "resnet50", "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "mobilenetv2", "mobilenet_v2"]:
        import timm
        timm_name = model_name
        if model_name == "mobilenetv2":
            timm_name = "mobilenetv2_100"
        model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
    elif model_name == "vit_small":
        import timm
        backbone = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0, dynamic_img_size=True)
        
        class ViTSmallClassifier(nn.Module):
            def __init__(self, backbone, num_classes):
                super().__init__()
                self.backbone = backbone
                self.fc = nn.Linear(384, num_classes)
                
            def forward(self, x):
                features = self.backbone(x)
                return self.fc(features)
                
        model = ViTSmallClassifier(backbone, num_classes)
        
        # Load self-supervised pre-trained backbone weights if specified in the config
        pretrained_backbone = config["model"].get("pretrained_backbone", "")
        if pretrained_backbone:
            import os
            if os.path.exists(pretrained_backbone):
                print(f"Loading pretrained SSL backbone weights from: {pretrained_backbone}")
                state_dict = torch.load(pretrained_backbone, map_location="cpu")
                # If checkpoint contains 'teacher' state dict (from train_ssl.py), extract backbone weights
                if isinstance(state_dict, dict) and "teacher" in state_dict:
                    teacher_state = state_dict["teacher"]
                    backbone_state = {}
                    for k, v in teacher_state.items():
                        if k.startswith("backbone."):
                            backbone_state[k.replace("backbone.", "")] = v
                    model.backbone.load_state_dict(backbone_state, strict=True)
                elif isinstance(state_dict, dict) and "state_dict" in state_dict:
                    model.load_state_dict(state_dict["state_dict"])
                else:
                    model.backbone.load_state_dict(state_dict, strict=True)
                print("Pretrained SSL backbone weights loaded successfully.")
            else:
                print(f"Warning: Pretrained backbone checkpoint not found at {pretrained_backbone}. Starting from random initialization.")
    elif model_name == "detnet59":
        from src.models.detnet.BasicModule import MydetNet59
        model = MydetNet59(pretrained=pretrained)
    elif model_name == "featherneta":
        from src.models.feathernet.FeatherNet import FeatherNetA
        input_size = config["data"]["input_size"]
        model = FeatherNetA(n_class=num_classes, input_size=input_size)
    elif model_name == "feathernetb":
        from src.models.feathernet.FeatherNet import FeatherNetB
        input_size = config["data"]["input_size"]
        model = FeatherNetB(n_class=num_classes, input_size=input_size)
    elif model_name == "aenet":
        from src.models.aenet.AENet import AENet
        model = AENet(num_classes=num_classes)
    elif model_name == "mobilenetv3_large":
        from src.models.MN3.MN3 import mobilenetv3_large
        mn3_config = config.get("model", {})
        model = mobilenetv3_large(
            width_mult=mn3_config.get("width_mult", 1.0),
            prob_dropout=mn3_config.get("prob_dropout", 0.2),
            type_dropout=mn3_config.get("type_dropout", "bernoulli"),
            prob_dropout_linear=mn3_config.get("prob_dropout_linear", 0.2),
            embeding_dim=mn3_config.get("embeding_dim", 128),
            mu=mn3_config.get("mu", 0.5),
            sigma=mn3_config.get("sigma", 0.3),
            theta=mn3_config.get("theta", 0.7),
            multi_heads=mn3_config.get("multi_heads", False)
        )
    elif model_name == "mobilenetv3_small":
        from src.models.MN3.MN3 import mobilenetv3_small
        mn3_config = config.get("model", {})
        model = mobilenetv3_small(
            width_mult=mn3_config.get("width_mult", 1.0),
            prob_dropout=mn3_config.get("prob_dropout", 0.2),
            type_dropout=mn3_config.get("type_dropout", "bernoulli"),
            prob_dropout_linear=mn3_config.get("prob_dropout_linear", 0.2),
            embeding_dim=mn3_config.get("embeding_dim", 128),
            mu=mn3_config.get("mu", 0.5),
            sigma=mn3_config.get("sigma", 0.3),
            theta=mn3_config.get("theta", 0.7),
            multi_heads=mn3_config.get("multi_heads", False)
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
        
    # Handle multi-GPU if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
        
    model = model.to(device)
    return model

def compute_loss(model, outputs, labels, batch, device, criterions, use_fourier):
    """
    Computes loss dynamically.
    If the model returns a tuple (classifier_outputs, fourier_outputs), 
    computes combined Classification + Fourier auxiliary loss.
    """
    if use_fourier and isinstance(outputs, tuple):
        cls_output, ft_output = outputs
        ft_target = batch[1].to(device) # ft_sample is the second item in batch
        
        loss_cls = criterions["cls"](cls_output, labels)
        loss_ft = criterions["ft"](ft_output, ft_target)
        
        # 0.5 weights as standard in MiniFASNet v2
        loss = 0.5 * loss_cls + 0.5 * loss_ft
        return loss, {"loss": loss.item(), "loss_cls": loss_cls.item(), "loss_ft": loss_ft.item()}
    else:
        # Standard classification loss
        cls_output = outputs[0] if isinstance(outputs, tuple) else outputs
        loss = criterions["cls"](cls_output, labels)
        return loss, {"loss": loss.item(), "loss_cls": loss.item()}

def train_one_epoch(model, dataloader, optimizer, criterions, device, use_fourier):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    # Track individual loss components
    loss_cls_accum = 0.0
    loss_ft_accum = 0.0
    
    progress_bar = tqdm(dataloader, desc="  Train", leave=False)
    for batch in progress_bar:
        # Batch unpacking
        if use_fourier:
            inputs, _, labels = batch
        else:
            inputs, labels = batch
            
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        # Calculate loss
        loss, loss_details = compute_loss(
            model, outputs, labels, batch, device, criterions, use_fourier
        )
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        loss_cls_accum += loss_details.get("loss_cls", 0.0) * inputs.size(0)
        loss_ft_accum += loss_details.get("loss_ft", 0.0) * inputs.size(0)
        
        # Calculate training accuracy (based on classification outputs)
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
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Val", leave=False):
            # For validation, we don't need FT maps, we just load images and labels
            # But the dataloader might still return them if use_fourier=True
            if len(batch) == 3:
                inputs, _, labels = batch
            else:
                inputs, labels = batch
                
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterions["cls"](outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels.append(labels.detach())
            all_preds.append(predicted.detach())
            
    val_loss = running_loss / len(dataloader.dataset)
    all_labels = torch.cat(all_labels).cpu().numpy()
    all_preds = torch.cat(all_preds).cpu().numpy()
    
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    apcer = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    bpcer = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    acer = (apcer + bpcer) / 2.0
    
    metrics = {
        "loss": val_loss,
        "acc": val_acc,
        "f1": val_f1,
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": acer
    }
    return metrics

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Override settings if command line args are passed
    if args.epochs:
        config["train"]["epochs"] = args.epochs
    if args.batch_size:
        config["train"]["batch_size"] = args.batch_size
    if args.lr:
        config["train"]["lr"] = args.lr
    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir
        
    # Environment settings
    device = torch.device(config["train"]["device"] if torch.cuda.is_available() and config["train"]["device"] == "cuda" else "cpu")
    print(f"Using device: {device}")
    
    # GPU optimizations

    # Detect Kaggle or Colab
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
    use_fourier = config["data"]["use_fourier"] and config["model"]["name"].lower() == "minifasv2"
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
            
    # 3. Setup criterions
    criterions = {
        "cls": nn.CrossEntropyLoss(),
        "ft": nn.MSELoss()
    }
    
    # 4. Setup Optimizer and Scheduler
    # Use parameters of module if wrapped in DataParallel
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
    monitor_metric = config["train"].get("early_stopping_monitor", "f1").lower()
    patience = config["train"].get("early_stopping_patience", 5)
    
    # Higher is better for accuracy and F1 score, lower is better for loss and ACER
    higher_is_better = monitor_metric in ["f1", "acc", "accuracy"]
    best_val_metric = -1e9 if higher_is_better else 1e9
    epochs_no_improve = 0
    
    print(f"Monitoring metric '{monitor_metric}' (higher is better: {higher_is_better}) for checkpoint saving and early stopping (patience={patience}).")
    
    start_epoch = 0
    
    # Check if we should resume from a checkpoint
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming training from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            
            # Load model weights
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                # Clean keys if model was trained with DataParallel but is being loaded without it
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
                
            # Load optimizer, scheduler, epoch, and metric states
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
            
            writer.add_scalar("Learning_Rate", optimizer.param_groups[0]["lr"], epoch)
            
            # Print epoch metrics
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Train F1: {train_f1*100:.2f}%")
            print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['acc'] * 100:.2f}% | Val F1: {val_metrics['f1'] * 100:.2f}%")
            print(f"  APCER: {val_metrics['apcer']*100:.2f}% | BPCER: {val_metrics['bpcer']*100:.2f}% | ACER: {val_metrics['acer']*100:.2f}%")
            
            # Determine current monitor metric value
            current_metric_val = val_metrics.get(monitor_metric, val_metrics["f1"])
            
            # Save checkpoints
            state = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_metric": best_val_metric,
                "monitor_metric": monitor_metric
            }
            
            # Save latest
            latest_path = os.path.join(config["train"]["save_dir"], f"{config['model']['name']}_latest.pth")
            save_checkpoint(state, latest_path)
            
            # Check if metric improved
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
                if monitor_metric != "loss":
                    print(f"  ★ New best model saved with Val {monitor_metric.upper()}: {best_val_metric*100:.2f}%")
                else:
                    print(f"  ★ New best model saved with Val LOSS: {best_val_metric:.4f}")
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
