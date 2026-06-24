import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import math

# Add project root to python path to resolve src.* imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import timm
from src.models.self_distill.model import DINOHead, MultiCropWrapper
from src.models.self_distill.loss import DINOLoss
from src.models.self_distill.dataset_ssl import DINODataAugmentation, AntiSpoofingSSLDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Self-Supervised Pretraining for ViT-Small (DINO)")
    parser.add_argument("--config", type=str, default="src/configs/self_distill_ssl.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--data-dir", type=str, default=None, help="Override data directory path")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

@torch.no_grad()
def ema_update(student, teacher, momentum):
    """
    EMA update from student to teacher.
    """
    student_state = student.state_dict()
    teacher_state = teacher.state_dict()
    for k in teacher_state.keys():
        if k in student_state:
            teacher_state[k].copy_(teacher_state[k] * momentum + student_state[k] * (1.0 - momentum))

def get_ema_momentum(epoch, total_epochs, base_momentum=0.996):
    """
    EMA momentum cosine scheduler: increases from base_momentum to 1.0.
    """
    return 1.0 - (1.0 - base_momentum) * (math.cos(math.pi * epoch / total_epochs) + 1.0) / 2.0

def save_checkpoint(state, filepath):
    temp_filepath = filepath + ".tmp"
    torch.save(state, temp_filepath)
    os.replace(temp_filepath, filepath)

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Override from command line if provided
    if args.epochs:
        config["train"]["epochs"] = args.epochs
    if args.batch_size:
        config["train"]["batch_size"] = args.batch_size
    if args.lr:
        config["train"]["lr"] = args.lr
    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir
        
    device = torch.device(config["train"]["device"] if torch.cuda.is_available() and config["train"]["device"] == "cuda" else "cpu")
    print(f"Using device: {device}")
    
    # Setup directories
    os.makedirs(config["train"]["save_dir"], exist_ok=True)
    os.makedirs(config["train"]["log_dir"], exist_ok=True)
    
    # 1. Dataset & DataLoader (Multi-Crop DINO)
    ssl_transform = DINODataAugmentation(
        global_crops_scale=tuple(config["data"]["global_crops_scale"]),
        local_crops_scale=tuple(config["data"]["local_crops_scale"]),
        local_crops_number=config["data"]["local_crops_number"],
        input_size=config["data"]["input_size"],
        local_size=config["data"]["local_size"]
    )
    
    dataset = AntiSpoofingSSLDataset(
        root_dir=config["data"]["data_dir"],
        split="train",
        transform=ssl_transform
    )
    
    num_workers = config["train"].get("num_workers", 4)
    if num_workers == "auto":
        num_workers = min(4, os.cpu_count() or 4)
        
    dataloader = DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers if len(dataset) > 0 else 0,
        persistent_workers=True if (num_workers > 0 and len(dataset) > 0) else False,
        drop_last=True # DINO loss requires complete batches for centering stability
    )
    
    # 2. Build Student and Teacher
    # ViT-Small features backbone
    student_backbone = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0, dynamic_img_size=True)
    teacher_backbone = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0, dynamic_img_size=True)
    
    # DINO MLP heads
    out_dim = config["model"]["out_dim"]
    student_head = DINOHead(in_dim=384, out_dim=out_dim)
    teacher_head = DINOHead(in_dim=384, out_dim=out_dim)
    
    student = MultiCropWrapper(student_backbone, student_head).to(device)
    teacher = MultiCropWrapper(teacher_backbone, teacher_head).to(device)
    
    # Synchronize teacher weights with student at start
    teacher.load_state_dict(student.state_dict())
    
    # Teacher does not require gradients
    for p in teacher.parameters():
        p.requires_grad = False
        
    print(f"Student & Teacher models built. Backbone: ViT-Small. Output Dim: {out_dim}")
    
    # 3. Setup Loss, Optimizer, and Scheduler
    epochs = config["train"]["epochs"]
    dino_loss = DINOLoss(
        out_dim=out_dim,
        ncrops=2 + config["data"]["local_crops_number"],
        warmup_teacher_temp=config["loss"]["warmup_teacher_temp"],
        teacher_temp=config["loss"]["teacher_temp"],
        warmup_teacher_temp_epochs=config["loss"]["warmup_teacher_temp_epochs"],
        epochs=epochs,
        student_temp=config["loss"]["student_temp"],
        center_momentum=config["loss"]["center_momentum"]
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"]
    )
    
    # Cosine annealing learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6
    )
    
    writer = SummaryWriter(log_dir=os.path.join(config["train"]["log_dir"], "self_distill_ssl"))
    
    best_loss = 1e9
    
    # 4. Training Loop
    print(f"Starting Self-Supervised pretraining for {epochs} epochs...")
    try:
        for epoch in range(epochs):
            student.train()
            teacher.train() # Teacher is kept in train to align batchnorm stats if used
            
            running_loss = 0.0
            ema_mom = get_ema_momentum(epoch, epochs, base_momentum=config["train"].get("ema_momentum", 0.996))
            
            progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=False)
            for batch_idx, crops in enumerate(progress_bar):
                # Move all crops to device
                crops = [crop.to(device, non_blocking=True) for crop in crops]
                
                # Student forward pass: processes all global and local crops
                student_output = student(crops)
                
                # Teacher forward pass: processes only global crops (first 2 crops in list)
                with torch.no_grad():
                    teacher_output = teacher(crops[:2])
                    
                loss = dino_loss(student_output, teacher_output, epoch)
                
                # Optimizer step
                optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients to prevent explosion (standard DINO practice)
                nn.utils.clip_grad_norm_(student.parameters(), max_norm=3.0)
                optimizer.step()
                
                # Update teacher weights using EMA
                ema_update(student, teacher, ema_mom)
                
                running_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item(), ema_mom=ema_mom)
                
            epoch_loss = running_loss / len(dataloader)
            lr_scheduler.step()
            
            # Log
            writer.add_scalar("Loss/train_ssl", epoch_loss, epoch)
            writer.add_scalar("LR/train_ssl", optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar("EMA_Momentum/train_ssl", ema_mom, epoch)
            
            print(f"Epoch [{epoch+1}/{epochs}] - Average Loss: {epoch_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            state = {
                "epoch": epoch + 1,
                "student": student.state_dict(),
                "teacher": teacher.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss
            }
            
            # Save latest checkpoint
            latest_path = os.path.join(config["train"]["save_dir"], "self_distill_ssl_latest.pth")
            save_checkpoint(state, latest_path)
            
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_path = os.path.join(config["train"]["save_dir"], "self_distill_ssl_best.pth")
                save_checkpoint(state, best_path)
                print(f"  ★ New best SSL model saved with Loss: {best_loss:.4f}")
                
        print("Self-Supervised training finished successfully!")
    except KeyboardInterrupt:
        print("\n[INFO] Pretraining interrupted by user. Gracefully shutting down...")
    finally:
        writer.close()

if __name__ == "__main__":
    main()
