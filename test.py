import os
import sys
import argparse
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

# Add project root to python path to resolve src.* imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.train import load_config, get_model
from src.evaluate import load_checkpoint
from src.data.dataset import SquarePad

def parse_args():
    parser = argparse.ArgumentParser(description="Test Anti-Spoofing Model on single image or folder")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model weights (.pth)")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--input", type=str, required=True, help="Path to an image file or a directory of images")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold (above is spoof)")
    parser.add_argument("--plot-path", type=str, default="evaluation_plots.png", help="Path to save the evaluation plots")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for directory testing")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for directory data loading")
    return parser.parse_args()

class InferenceDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        file_path = self.image_paths[idx]
        img = cv2.imread(file_path)
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tensor_img = self.transform(pil_img)
        
        # Determine ground truth from path parts
        gt_label = -1
        has_gt = False
        path_parts = file_path.lower().replace('\\', '/').split('/')
        if 'live' in path_parts or 'real' in path_parts:
            gt_label = 0
            has_gt = True
        elif 'spoof' in path_parts or 'fake' in path_parts:
            gt_label = 1
            has_gt = True
            
        return tensor_img, file_path, gt_label, has_gt

def predict_single(model, image_path, config, transform, device, threshold):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None
    
    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # Apply standard preprocessing
    tensor_img = transform(pil_img)  # (3, H, W)
    tensor_img = tensor_img.unsqueeze(0)  # Add batch dimension: (1, 3, H, W)
    
    # Handle sequential models
    model_name = config["model"]["name"].lower()
    is_sequence = "lstm" in model_name or "multi_frame" in model_name
    
    if is_sequence:
        len_seq = config["model"].get("len_seq", 5 if "multi" in model_name else 1)
        # Shape: (1, T, 3, H, W) by repeating the single image T times
        tensor_img = tensor_img.unsqueeze(1).repeat(1, len_seq, 1, 1, 1)
        
    tensor_img = tensor_img.to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(tensor_img)
        # If output is a tuple (e.g. from Fourier models during evaluation), take the first item
        if isinstance(outputs, tuple):
            outputs = outputs[0]
            
        probs = torch.softmax(outputs, dim=1)
        prob_spoof = probs[0, 1].item()
        prob_real = probs[0, 0].item()
        
    # Classify based on threshold
    prediction = "Spoof/Fake" if prob_spoof >= threshold else "Real/Live"
    confidence = prob_spoof if prob_spoof >= threshold else prob_real
    
    return {
        "prediction": prediction,
        "prob_spoof": prob_spoof,
        "prob_real": prob_real,
        "confidence": confidence
    }

def main():
    args = parse_args()
    config = load_config(args.config)
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 1. Load Model
    model = get_model(config, device)
    load_checkpoint(model, args.model_path, device)
    model.eval()
    
    # 2. Setup preprocessing transforms
    input_size = config["data"]["input_size"]
    transform = T.Compose([
        SquarePad(),
        T.Resize((input_size, input_size)),
        T.ToTensor(),
    ])
    
    # 3. Process Input
    input_path = args.input
    if os.path.isfile(input_path):
        # Single image testing
        print(f"\nTesting single image: {input_path}")
        result = predict_single(model, input_path, config, transform, device, args.threshold)
        if result:
            print("=" * 40)
            print(f"Result: {result['prediction']}")
            print(f"Confidence: {result['confidence']*100:.2f}%")
            print("-" * 40)
            print(f"Probability Real/Live: {result['prob_real']:.4f}")
            print(f"Probability Spoof/Fake: {result['prob_spoof']:.4f}")
            print("=" * 40)
            
    elif os.path.isdir(input_path):
        # Directory of images testing (find recursively)
        print(f"\nTesting directory: {input_path}")
        image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        
        # Find all files recursively using os.walk
        all_image_paths = []
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if file.lower().endswith(image_extensions):
                    all_image_paths.append(os.path.join(root, file))
                    
        if not all_image_paths:
            print(f"No image files found recursively in {input_path}")
            return
            
        print(f"Found {len(all_image_paths)} images. Setting up DataLoader...")
        
        # Determine sequence parameters
        model_name = config["model"]["name"].lower()
        is_sequence = "lstm" in model_name or "multi_frame" in model_name
        len_seq = config["model"].get("len_seq", 5 if "multi" in model_name else 1)

        dataset = InferenceDataset(sorted(all_image_paths), transform)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if device.type == "cuda" else False
        )
        
        real_count = 0
        spoof_count = 0
        correct_count = 0
        has_ground_truth = False
        
        y_true = []
        y_score = []
        
        print(f"Running batch inference (batch_size={args.batch_size}, num_workers={args.num_workers})...")
        from tqdm import tqdm
        
        with torch.no_grad():
            for batch_tensors, batch_paths, batch_gt, batch_has_gt in tqdm(dataloader, desc="Evaluating Directory"):
                if is_sequence:
                    batch_tensors = batch_tensors.unsqueeze(1).repeat(1, len_seq, 1, 1, 1)
                
                batch_tensors = batch_tensors.to(device)
                outputs = model(batch_tensors)
                
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                    
                probs = torch.softmax(outputs, dim=1)
                prob_spoof = probs[:, 1].cpu().numpy()
                prob_real = probs[:, 0].cpu().numpy()
                
                for idx in range(len(batch_paths)):
                    p_spoof = float(prob_spoof[idx])
                    p_real = float(prob_real[idx])
                    gt_idx = int(batch_gt[idx])
                    has_gt = bool(batch_has_gt[idx])
                    
                    prediction = "Spoof/Fake" if p_spoof >= args.threshold else "Real/Live"
                    
                    if prediction == "Real/Live":
                        real_count += 1
                    else:
                        spoof_count += 1
                        
                    if has_gt:
                        has_ground_truth = True
                        y_true.append(gt_idx)
                        y_score.append(p_spoof)
                        
                        # Compare prediction with gt
                        gt_str = "Real/Live" if gt_idx == 0 else "Spoof/Fake"
                        if prediction == gt_str:
                            correct_count += 1
                            
        print("-" * 105)
        print(f"Summary:")
        print(f"Total processed: {len(all_image_paths)}")
        print(f"Real/Live:       {real_count} ({real_count/len(all_image_paths)*100:.1f}%)")
        print(f"Spoof/Fake:      {spoof_count} ({spoof_count/len(all_image_paths)*100:.1f}%)")
        
        if has_ground_truth:
            accuracy = correct_count / len(y_true) * 100
            print(f"Accuracy:        {correct_count}/{len(y_true)} ({accuracy:.2f}%)")
            
            # Print ROC & APCER/BPCER curves
            if len(y_true) > 1:
                from sklearn.metrics import roc_curve, auc, confusion_matrix
                y_true = np.array(y_true)
                y_score = np.array(y_score)
                
                # Compute ROC
                fpr, tpr, thresholds = roc_curve(y_true, y_score)
                fnr = 1 - tpr  # APCER is FNR (False Negative Rate)
                roc_auc = auc(fpr, tpr)
                
                # Compute EER
                idx = np.nanargmin(np.abs(fpr - fnr))
                eer = (fpr[idx] + fnr[idx]) / 2.0
                eer_threshold = thresholds[idx]
                
                # Compute APCER & BPCER at specified threshold (args.threshold)
                y_pred = (y_score >= args.threshold).astype(int)
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
                apcer_at_thresh = fn / (fn + tp) if (fn + tp) > 0 else 0.0
                bpcer_at_thresh = fp / (tn + fp) if (tn + fp) > 0 else 0.0
                
                # Compute APCER @ BPCER = 1%
                # 1. Exact value by interpolation (standard ROC analysis)
                apcer_at_bpcer_1pct_interp = 1.0 - float(np.interp(0.01, fpr, tpr))
                # 2. Conservative value where BPCER <= 1% (ISO standard, thresholding based)
                indices_bpcer = np.where(fpr <= 0.01)[0]
                if len(indices_bpcer) > 0:
                    idx_bpcer = indices_bpcer[-1]
                    apcer_at_bpcer_1pct_cons = 1.0 - tpr[idx_bpcer]
                    thresh_at_bpcer_1pct = thresholds[idx_bpcer]
                else:
                    apcer_at_bpcer_1pct_cons = 1.0
                    thresh_at_bpcer_1pct = 1.0
                    
                # Compute BPCER @ APCER = 1%
                # 1. Exact value by interpolation (standard ROC analysis)
                bpcer_at_apcer_1pct_interp = float(np.interp(0.01, fnr[::-1], fpr[::-1]))
                # 2. Conservative value where APCER <= 1% (ISO standard, thresholding based)
                indices_apcer = np.where(fnr <= 0.01)[0]
                if len(indices_apcer) > 0:
                    idx_apcer = indices_apcer[0]
                    bpcer_at_apcer_1pct_cons = fpr[idx_apcer]
                    thresh_at_apcer_1pct = thresholds[idx_apcer]
                else:
                    bpcer_at_apcer_1pct_cons = 1.0
                    thresh_at_apcer_1pct = 0.0

                print(f"ROC-AUC Score:   {roc_auc:.4f}")
                print(f"EER (Equal Error Rate): {eer*100:.2f}% (at threshold {eer_threshold:.4f})")
                print("-" * 50)
                print(f"Metrics at Threshold {args.threshold:.4f}:")
                print(f"  APCER: {apcer_at_thresh*100:.2f}% (Spoof as Real)")
                print(f"  BPCER: {bpcer_at_thresh*100:.2f}% (Real as Spoof)")
                print(f"  ACER:  {(apcer_at_thresh + bpcer_at_thresh)*50:.2f}%")
                print("-" * 50)
                print(f"Metrics at BPCER (FPR) = 1%:")
                print(f"  APCER @ BPCER=1% (Interp): {apcer_at_bpcer_1pct_interp*100:.2f}% (TPR: {(1.0-apcer_at_bpcer_1pct_interp)*100:.2f}%)")
                print(f"  APCER @ BPCER=1% (Cons):   {apcer_at_bpcer_1pct_cons*100:.2f}% (at threshold {thresh_at_bpcer_1pct:.4f})")
                print("-" * 50)
                print(f"Metrics at APCER (FNR) = 1%:")
                print(f"  BPCER @ APCER=1% (Interp): {bpcer_at_apcer_1pct_interp*100:.2f}%")
                print(f"  BPCER @ APCER=1% (Cons):   {bpcer_at_apcer_1pct_cons*100:.2f}% (at threshold {thresh_at_apcer_1pct:.4f})")
                                
                # Generate plots using matplotlib
                try:
                    import matplotlib
                    matplotlib.use('Agg')  # Headless mode to avoid Display errors
                    import matplotlib.pyplot as plt
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                    
                    # Plot 1: ROC Curve
                    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
                    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    ax1.set_xlim([0.0, 1.0])
                    ax1.set_ylim([0.0, 1.05])
                    ax1.set_xlabel('False Positive Rate (BPCER)')
                    ax1.set_ylabel('True Positive Rate (1 - APCER)')
                    ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
                    ax1.legend(loc="lower right")
                    ax1.grid(True, linestyle='--', alpha=0.5)
                    
                    # Plot 2: APCER and BPCER vs Threshold
                    # Filter thresholds <= 1.0 for visualization
                    valid_indices = thresholds <= 1.0
                    plot_thresholds = thresholds[valid_indices]
                    plot_apcer = fnr[valid_indices]
                    plot_bpcer = fpr[valid_indices]
                    
                    ax2.plot(plot_thresholds, plot_apcer * 100, color='red', lw=2, label='APCER (FNR)')
                    ax2.plot(plot_thresholds, plot_bpcer * 100, color='blue', lw=2, label='BPCER (FPR)')
                    ax2.scatter(eer_threshold, eer * 100, color='green', s=100, zorder=5,
                                label=f'EER = {eer*100:.2f}% (Thresh = {eer_threshold:.3f})')
                    
                    ax2.set_xlim([0.0, 1.0])
                    ax2.set_ylim([0.0, 105.0])
                    ax2.set_xlabel('Threshold')
                    ax2.set_ylabel('Error Rate (%)')
                    ax2.set_title('APCER / BPCER vs. Classification Threshold')
                    ax2.legend(loc="upper right")
                    ax2.grid(True, linestyle='--', alpha=0.5)
                    
                    plt.tight_layout()
                    plt.savefig(args.plot_path)
                    print(f"Saved evaluation plots to: {args.plot_path}")
                    plt.close()
                except Exception as e:
                    print(f"Warning: Matplotlib plot generation failed. Error: {e}")
                    
        print("=" * 105)
    else:
        print(f"Error: Input path {input_path} is neither a file nor a directory.")

if __name__ == "__main__":
    main()
    