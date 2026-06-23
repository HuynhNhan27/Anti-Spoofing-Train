import os
import sys
# Add project root to python path to resolve src.* imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve
from tqdm import tqdm

from src.data.dataset import get_dataloader
from src.train import get_model

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Anti-Spoofing Models")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model weights (.pth)")
    parser.add_argument("--config", type=str, default="src/configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract state_dict (handles cases where the complete state dict dict is saved)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
        
    # Clean keys if model was trained with DataParallel but is being loaded without it
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
        
    # If our current model is wrapped in DataParallel, add "module." prefix
    if isinstance(model, nn.DataParallel):
        dp_state_dict = {}
        for k, v in new_state_dict.items():
            dp_state_dict[f"module.{k}"] = v
        new_state_dict = dp_state_dict
        
    model.load_state_dict(new_state_dict, strict=True)
    print(f"Loaded weights successfully from {checkpoint_path}")

def calculate_eer(labels, spoof_scores):
    """
    Calculate Equal Error Rate (EER) and the corresponding threshold.
    Note:
    - Labels: 0 for Real/Live, 1 for Spoof/Fake.
    - spoof_scores: Probability of class 1 (Spoof).
    """
    # BPCER (Bona Fide Presentation Classification Error Rate) = FPR (False Positive Rate)
    # APCER (Attack Presentation Classification Error Rate) = FNR (False Negative Rate) = 1 - TPR (True Positive Rate)
    fpr, tpr, thresholds = roc_curve(labels, spoof_scores, pos_label=1)
    fnr = 1.0 - tpr
    
    # Find EER index where BPCER (fpr) and APCER (fnr) are closest
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    eer_threshold = thresholds[idx]
    
    return eer, eer_threshold

def evaluate_model(model, dataloader, device):
    model.eval()
    all_scores = []  # Store probability of spoof (class 1)
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if len(batch) == 3:
                inputs, _, labels = batch
            else:
                inputs, labels = batch
                
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Compute probabilities using softmax
            probs = torch.softmax(outputs, dim=1)
            # Probability of spoof class (1)
            spoof_prob = probs[:, 1].cpu().numpy()
            
            all_scores.extend(spoof_prob)
            all_labels.extend(labels.numpy())
            
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # Calculate standard accuracy using 0.5 threshold
    predictions = (all_scores >= 0.5).astype(int)
    correct = np.sum(predictions == all_labels)
    accuracy = correct / len(all_labels)
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate error rates at default 0.5 threshold
    # APCER (Attack Presentation Classification Error Rate) = Spoofs classified as Real
    apcer = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    # BPCER (Bona Fide Presentation Classification Error Rate) = Reals classified as Spoof
    bpcer = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    hter = (apcer + bpcer) / 2.0
    
    # Calculate EER (Equal Error Rate) and its threshold
    eer, eer_threshold = calculate_eer(all_labels, all_scores)
    
    # Calculate APCER and BPCER at the EER threshold
    eer_predictions = (all_scores >= eer_threshold).astype(int)
    cm_eer = confusion_matrix(all_labels, eer_predictions, labels=[0, 1])
    tn_e, fp_e, fn_e, tp_e = cm_eer.ravel()
    apcer_at_eer = fn_e / (fn_e + tp_e) if (fn_e + tp_e) > 0 else 0.0
    bpcer_at_eer = fp_e / (tn_e + fp_e) if (tn_e + fp_e) > 0 else 0.0
    
    results = {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "apcer_0.5": apcer,
        "bpcer_0.5": bpcer,
        "hter_0.5": hter,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "apcer_at_eer": apcer_at_eer,
        "bpcer_at_eer": bpcer_at_eer
    }
    
    return results

def main():
    args = parse_args()
    config = load_config(args.config)
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Evaluating on device: {device}")
    
    # 1. Get Model
    model = get_model(config, device)
    
    # 2. Load Checkpoint
    load_checkpoint(model, args.model_path, device)
    
    # 3. Load Dataloader for test split
    # For evaluation, we only classify so use_fourier is false
    test_loader = get_dataloader(
        data_dir=config["data"]["data_dir"],
        split="test",
        batch_size=config["train"]["batch_size"],
        input_size=config["data"]["input_size"],
        use_fourier=False,
        is_train=False
    )
    
    if len(test_loader.dataset) == 0:
        print("Error: Test dataset is empty. Cannot perform evaluation.")
        return
        
    # 4. Evaluate
    results = evaluate_model(model, test_loader, device)
    
    # 5. Print Metrics
    print("\n" + "="*50)
    print("EVALUATION METRICS REPORT")
    print("="*50)
    print(f"Test Accuracy (threshold 0.5): {results['accuracy']*100:.2f}%")
    print(f"Confusion Matrix (threshold 0.5):\n{results['confusion_matrix']}")
    print(f"APCER (threshold 0.5): {results['apcer_0.5']*100:.2f}% (Spoof classified as Real)")
    print(f"BPCER (threshold 0.5): {results['bpcer_0.5']*100:.2f}% (Real classified as Spoof)")
    print(f"HTER  (threshold 0.5): {results['hter_0.5']*100:.2f}%")
    print("-" * 50)
    print(f"EER (Equal Error Rate): {results['eer']*100:.2f}%")
    print(f"EER Threshold: {results['eer_threshold']:.4f}")
    print(f"APCER @ EER: {results['apcer_at_eer']*100:.2f}%")
    print(f"BPCER @ EER: {results['bpcer_at_eer']*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()
