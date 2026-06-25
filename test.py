import os
import sys
import argparse
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

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
    return parser.parse_args()

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
            
        print(f"Found {len(all_image_paths)} images. Running inference...")
        
        real_count = 0
        spoof_count = 0
        correct_count = 0
        has_ground_truth = False
        
        y_true = []
        y_score = []
        
        # Column headers
        print(f"{'Relative Path':<50} | {'Prediction':<12} | {'Ground Truth':<12} | {'Prob Real':<10} | {'Prob Spoof':<10}")
        print("-" * 105)
        
        for file_path in sorted(all_image_paths):
            result = predict_single(model, file_path, config, transform, device, args.threshold)
            if result:
                # Get path relative to the input folder
                rel_path = os.path.relpath(file_path, input_path)
                
                # Determine ground truth from path
                gt = None
                path_parts = rel_path.lower().split(os.sep)
                if 'live' in path_parts or 'real' in path_parts:
                    gt = "Real/Live"
                    has_ground_truth = True
                    y_true.append(0)
                    y_score.append(result["prob_spoof"])
                elif 'spoof' in path_parts or 'fake' in path_parts:
                    gt = "Spoof/Fake"
                    has_ground_truth = True
                    y_true.append(1)
                    y_score.append(result["prob_spoof"])
                
                # Count stats
                if result["prediction"] == "Real/Live":
                    real_count += 1
                else:
                    spoof_count += 1
                    
                # Accuracy tracking
                is_correct_str = ""
                if gt is not None:
                    if result["prediction"] == gt:
                        correct_count += 1
                        is_correct_str = "✓"
                    else:
                        is_correct_str = "✗"
                
                # gt_display = gt if gt is not None else "Unknown"
                # pred_display = f"{result['prediction']} {is_correct_str}".strip()
                
                # Print row (truncate relative path if too long)
                # display_path = rel_path if len(rel_path) <= 48 else "..." + rel_path[-45:]
                # print(f"{display_path:<50} | {pred_display:<12} | {gt_display:<12} | {result['prob_real']:.4f}     | {result['prob_spoof']:.4f}")
                
        print("-" * 105)
        print(f"Summary:")
        print(f"Total processed: {len(all_image_paths)}")
        print(f"Real/Live:       {real_count} ({real_count/len(all_image_paths)*100:.1f}%)")
        print(f"Spoof/Fake:      {spoof_count} ({spoof_count/len(all_image_paths)*100:.1f}%)")
        
        if has_ground_truth:
            accuracy = correct_count / len(all_image_paths) * 100
            print(f"Accuracy:        {correct_count}/{len(all_image_paths)} ({accuracy:.2f}%)")
            
            # Print ROC & APCER/BPCER curves
            if len(y_true) > 1:
                from sklearn.metrics import roc_curve, auc
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
                
                print(f"ROC-AUC Score:   {roc_auc:.4f}")
                print(f"EER (Equal Error Rate): {eer*100:.2f}% (at threshold {eer_threshold:.4f})")
                
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
