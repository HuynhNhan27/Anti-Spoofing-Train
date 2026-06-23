import argparse
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.dataset import get_dataloader

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Anti-Spoofing Model Training")
    parser.add_argument("--config", type=str, default="src/configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    return parser.parse_args()

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
    correct = 0
    total = 0
    
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
            
        inputs, labels = inputs.to(device), labels.to(device)
        
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
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        progress_bar.set_postfix(loss=loss.item(), acc=100.0 * correct / total)
        
    epoch_loss = running_loss / total
    epoch_loss_cls = loss_cls_accum / total
    epoch_loss_ft = loss_ft_accum / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_loss_cls, epoch_loss_ft, epoch_acc

def validate(model, dataloader, criterions, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Val", leave=False):
            # For validation, we don't need FT maps, we just load images and labels
            # But the dataloader might still return them if use_fourier=True
            if len(batch) == 3:
                inputs, _, labels = batch
            else:
                inputs, labels = batch
                
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass: MultiFTNet in eval() mode returns only logits (not a tuple)
            outputs = model(inputs)
            
            # Loss calculation
            loss = criterions["cls"](outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc

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
        
    # Environment settings
    device = torch.device(config["train"]["device"] if torch.cuda.is_available() and config["train"]["device"] == "cuda" else "cpu")
    print(f"Using device: {device}")
    
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
        is_train=True
    )
    val_loader = get_dataloader(
        data_dir=config["data"]["data_dir"],
        split="val",
        batch_size=config["train"]["batch_size"],
        input_size=config["data"]["input_size"],
        use_fourier=use_fourier, # Pass same flag, but validation loop only runs liveness classifier
        is_train=False
    )
    
    # 2. Get Model
    model = get_model(config, device)
    
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
    
    # 6. Training loop
    best_val_acc = 0.0
    print(f"Starting training for model: {config['model']['name']} for {config['train']['epochs']} epochs...")
    
    for epoch in range(config["train"]["epochs"]):
        print(f"Epoch [{epoch+1}/{config['train']['epochs']}]")
        
        # Train
        train_loss, train_loss_cls, train_loss_ft, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterions, device, use_fourier
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterions, device)
        
        scheduler.step()
        
        # Log to TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/train_cls", train_loss_cls, epoch)
        if use_fourier:
            writer.add_scalar("Loss/train_ft", train_loss_ft, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]["lr"], epoch)
        
        # Print epoch metrics
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        
        # Save checkpoints
        state = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc": best_val_acc
        }
        
        # Save latest
        latest_path = os.path.join(config["train"]["save_dir"], f"{config['model']['name']}_latest.pth")
        torch.save(state, latest_path)
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(config["train"]["save_dir"], f"{config['model']['name']}_best.pth")
            torch.save(state, best_path)
            print(f"  ★ New best model saved with Val Acc: {val_acc*100:.2f}%")
            
    writer.close()
    print("Training finished successfully!")

if __name__ == "__main__":
    main()
