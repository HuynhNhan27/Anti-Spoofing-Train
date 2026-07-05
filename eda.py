import os
import sys
import time
import random
import argparse
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Style settings for clean, high-quality plots
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 150
})

def parse_arguments():
    parser = argparse.ArgumentParser(description="Exploratory Data Analysis (EDA) on Anti-Spoofing Image Dataset")
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="data/celeba-spoof-cropped-sampled/dataset",
        help="Path to the dataset directory (containing train/val/test splits)"
    )
    parser.add_argument(
        "--max-samples", 
        type=int, 
        default=25000,
        help="Maximum samples to analyze per class per split to optimize runtime. Use -1 for all."
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="eda_results",
        help="Directory where EDA reports and plots will be saved"
    )
    parser.add_argument(
        "--num-workers", 
        type=int, 
        default=min(8, cpu_count()),
        help="Number of parallel processes to use for parsing images"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for sample selection consistency"
    )
    return parser.parse_args()

def analyze_single_image(args):
    """
    Worker function to load and extract image characteristics.
    """
    img_path, split, class_name, label = args
    try:
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            return None
        
        h, w, c = img.shape
        aspect_ratio = w / h if h > 0 else 0
        area = w * h
        
        # Convert to Grayscale for texture & contrast analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Brightness: Mean intensity
        brightness = np.mean(gray)
        
        # 2. Contrast: RMS Contrast (Standard deviation of pixel intensities)
        contrast = np.std(gray)
        
        # 3. Sharpness / Focus: Laplacian Variance
        # Higher variance indicates more sharp/focused edges, lower indicates blurry image
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Convert to HSV for color space analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_mean = np.mean(hsv[:, :, 0])
        s_mean = np.mean(hsv[:, :, 1])
        v_mean = np.mean(hsv[:, :, 2])
        
        # Mean RGB values
        # OpenCV loads in BGR format
        b_mean = np.mean(img[:, :, 0])
        g_mean = np.mean(img[:, :, 1])
        r_mean = np.mean(img[:, :, 2])
        
        # Fourier Transform (FT) magnitude map for frequency domain analysis
        # Resize to fixed size to keep FFT maps comparable
        fft_size = 128
        gray_resized = cv2.resize(gray, (fft_size, fft_size))
        f = np.fft.fft2(gray_resized)
        fshift = np.fft.fftshift(f)
        fft_magnitude = np.log(np.abs(fshift) + 1)
        
        return {
            "path": img_path,
            "filename": os.path.basename(img_path),
            "split": split,
            "class": class_name,
            "label": label,
            "width": w,
            "height": h,
            "aspect_ratio": aspect_ratio,
            "resolution": area,
            "brightness": brightness,
            "contrast": contrast,
            "sharpness": sharpness,
            "hue_mean": h_mean,
            "sat_mean": s_mean,
            "val_mean": v_mean,
            "r_mean": r_mean,
            "g_mean": g_mean,
            "b_mean": b_mean,
            "fft_magnitude": fft_magnitude
        }
    except Exception as e:
        # Silently skip corrupted files but log their path
        return {"error": f"Failed to process {img_path}: {str(e)}"}

def main():
    args = parse_arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Dataset root: {args.data_dir}")
    print(f"Max samples per class/split: {args.max_samples if args.max_samples > 0 else 'All'}")
    print(f"Workers count: {args.num_workers}")
    
    # 1. Scan directory structure
    splits = ["train"]
    classes = ["live", "spoof"]
    class_map = {"live": 0, "spoof": 1}
    
    all_image_jobs = []
    dataset_counts = {}
    
    for split in splits:
        split_dir = os.path.join(args.data_dir, split)
        dataset_counts[split] = {}
        if not os.path.exists(split_dir):
            print(f"Warning: Split directory {split_dir} does not exist. Skipping.")
            continue
            
        for class_name in classes:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                dataset_counts[split][class_name] = 0
                continue
            
            # Find all image paths
            img_paths = []
            for root, _, files in os.walk(class_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        img_paths.append(os.path.join(root, file))
            
            dataset_counts[split][class_name] = len(img_paths)
            
            # Sample paths if requested
            if args.max_samples > 0 and len(img_paths) > args.max_samples:
                sampled_paths = random.sample(img_paths, args.max_samples)
            else:
                sampled_paths = img_paths
                
            label = class_map[class_name]
            for p in sampled_paths:
                all_image_jobs.append((p, split, class_name, label))
                
    # Print class balance table
    print("\nDataset Class Distributions (Total Found in Directories):")
    print("-" * 50)
    print(f"{'Split':<10} | {'Live':<12} | {'Spoof':<12} | {'Total':<12}")
    print("-" * 50)
    for split in splits:
        live_c = dataset_counts.get(split, {}).get("live", 0)
        spoof_c = dataset_counts.get(split, {}).get("spoof", 0)
        print(f"{split:<10} | {live_c:<12,} | {spoof_c:<12,} | {live_c + spoof_c:<12,}")
    print("-" * 50)
    
    if not all_image_jobs:
        print("Error: No images found. Check your --data-dir argument.")
        sys.exit(1)
        
    print(f"\nProcessing/Analyzing a sample of {len(all_image_jobs):,} images using {args.num_workers} processes...")
    
    # Process images in parallel
    results = []
    errors = []
    
    with Pool(processes=args.num_workers) as pool:
        for res in tqdm(pool.imap_unordered(analyze_single_image, all_image_jobs), total=len(all_image_jobs)):
            if res is None:
                continue
            if "error" in res:
                errors.append(res["error"])
            else:
                results.append(res)
                
    if errors:
        print(f"Skipped {len(errors)} corrupted or unreadable images.")
        # Log first 5 errors
        for err in errors[:5]:
            print(f"  - {err}")
            
    if not results:
        print("Error: No image was successfully analyzed.")
        sys.exit(1)
        
    # Create pandas DataFrame
    df = pd.DataFrame(results)
    
    # Extract Fourier magnitudes for average maps
    fft_maps = {
        "live": [],
        "spoof": []
    }
    
    # Clean df columns: we don't want 'fft_magnitude' column to sit in the csv dataframe as it contains 2D arrays
    # We will extract them and drop from df
    for r in results:
        if r["fft_magnitude"] is not None:
            fft_maps[r["class"]].append(r["fft_magnitude"])
            
    df_clean = df.drop(columns=["fft_magnitude"])
    df_clean.to_csv(os.path.join(args.output_dir, "image_metrics.csv"), index=False)
    print(f"Saved raw image metrics to: {os.path.join(args.output_dir, 'image_metrics.csv')}")
    
    # 2. Compute summary statistics
    metrics_to_summary = ["width", "height", "aspect_ratio", "resolution", "brightness", "contrast", "sharpness", "hue_mean", "sat_mean", "val_mean"]
    
    print("\n=== Summary Statistics (Live vs Spoof) ===")
    summary_list = []
    for metric in metrics_to_summary:
        grouped = df_clean.groupby(["class"])[metric].agg(["mean", "std", "min", "median", "max"])
        grouped["metric"] = metric
        summary_list.append(grouped.reset_index())
        
        # Print summaries to stdout in a pretty format
        print(f"\nMetric: {metric.upper()}")
        print("-" * 75)
        print(f"{'Class':<8} | {'Mean':<12} | {'Std':<12} | {'Min':<12} | {'Median':<12} | {'Max':<12}")
        print("-" * 75)
        for idx, row in grouped.iterrows():
            print(f"{idx:<8} | {row['mean']:<12.3f} | {row['std']:<12.3f} | {row['min']:<12.3f} | {row['median']:<12.3f} | {row['max']:<12.3f}")
        print("-" * 75)
        
    summary_df = pd.concat(summary_list, ignore_index=True)
    summary_df.to_csv(os.path.join(args.output_dir, "summary_statistics.csv"), index=False)
    print(f"Saved aggregated statistics to: {os.path.join(args.output_dir, 'summary_statistics.csv')}")
    
    # Write a quick markdown summary file
    write_markdown_report(args.output_dir, dataset_counts, df_clean)
    
    # 3. Create visualizations
    print("\nGenerating EDA plots...")
    
    # Plot 1: Image Dimensions Scatter and Histograms
    plot_dimensions(df_clean, args.output_dir)
    
    # Plot 2: Brightness & Contrast Distribution Comparison
    plot_brightness_contrast(df_clean, args.output_dir)
    
    # Plot 3: Sharpness / Blurriness Comparison (Laplacian Variance)
    plot_sharpness(df_clean, args.output_dir)
    
    # Plot 4: Saturation & Value distributions
    plot_color_distributions(df_clean, args.output_dir)
    
    # Plot 5: Average Fourier Transform Power Spectrums (Frequency analysis)
    plot_fourier_analysis(fft_maps, args.output_dir)
    
    # Plot 6: Combined EDA Dashboard
    plot_dashboard(df_clean, fft_maps, args.output_dir)
    
    print(f"\n=== EDA Complete! Results saved to '{args.output_dir}/' directory ===")
    print("Files generated:")
    print(f"  1. Metrics CSV:      {args.output_dir}/image_metrics.csv")
    print(f"  2. Statistics CSV:   {args.output_dir}/summary_statistics.csv")
    print(f"  3. Markdown Report:  {args.output_dir}/eda_report.md")
    print(f"  4. Dimension Plot:   {args.output_dir}/dimensions_distribution.png")
    print(f"  5. Intensity Plot:   {args.output_dir}/brightness_contrast.png")
    print(f"  6. Sharpness Plot:   {args.output_dir}/sharpness_distribution.png")
    print(f"  7. HSV Color Plot:   {args.output_dir}/color_spaces.png")
    print(f"  8. Fourier Plot:     {args.output_dir}/fourier_analysis.png")
    print(f"  9. Complete Dash:    {args.output_dir}/eda_dashboard.png")

def plot_dimensions(df, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # W vs H scatter
    colors = {"live": "#2ca02c", "spoof": "#d62728"}
    for name, group in df.groupby("class"):
        axes[0].scatter(group["width"], group["height"], alpha=0.4, label=name, color=colors[name], edgecolors='none', s=20)
    axes[0].set_xlabel("Width (pixels)")
    axes[0].set_ylabel("Height (pixels)")
    axes[0].set_title("Image Dimensions Scatter")
    axes[0].legend()
    
    # Aspect Ratio density
    for name, group in df.groupby("class"):
        axes[1].hist(group["aspect_ratio"], bins=30, alpha=0.5, label=name, color=colors[name], density=True)
    axes[1].set_xlabel("Aspect Ratio (Width / Height)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Aspect Ratio Distribution")
    axes[1].legend()
    
    # Resolution Area distribution
    for name, group in df.groupby("class"):
        # Log scale for resolution since it might span multiple magnitudes
        axes[2].hist(np.log10(group["resolution"]), bins=30, alpha=0.5, label=name, color=colors[name], density=True)
    axes[2].set_xlabel("Resolution (Log10 scale, pixels)")
    axes[2].set_ylabel("Density")
    axes[2].set_title("Resolution (Area) Distribution")
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dimensions_distribution.png"), bbox_inches='tight')
    plt.close()

def plot_brightness_contrast(df, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"live": "#2ca02c", "spoof": "#d62728"}
    
    # Brightness (Mean Grayscale)
    for name, group in df.groupby("class"):
        axes[0].hist(group["brightness"], bins=50, alpha=0.5, label=name, color=colors[name], density=True)
    axes[0].set_xlabel("Grayscale Brightness (0-255)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Mean Brightness Distribution")
    axes[0].legend()
    
    # Contrast (Standard Deviation)
    for name, group in df.groupby("class"):
        axes[1].hist(group["contrast"], bins=50, alpha=0.5, label=name, color=colors[name], density=True)
    axes[1].set_xlabel("RMS Contrast (Std of intensity)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Contrast Distribution")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "brightness_contrast.png"), bbox_inches='tight')
    plt.close()

def plot_sharpness(df, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"live": "#2ca02c", "spoof": "#d62728"}
    
    # Sharpness density histogram
    for name, group in df.groupby("class"):
        # Laplacian variance has a long tail, we use log scale for visual clarity
        log_sharp = np.log10(group["sharpness"] + 1)
        axes[0].hist(log_sharp, bins=50, alpha=0.5, label=name, color=colors[name], density=True)
    axes[0].set_xlabel("Sharpness Log10(Laplacian Var + 1)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Image Sharpness (Log Scale)")
    axes[0].legend()
    
    # Box plot
    data_to_plot = []
    labels = []
    for name, group in df.groupby("class"):
        data_to_plot.append(np.log10(group["sharpness"] + 1))
        labels.append(name.capitalize())
        
    bp = axes[1].boxplot(data_to_plot, patch_artist=True, labels=labels)
    # Style the box plot
    for patch, color in zip(bp['boxes'], [colors['live'], colors['spoof']]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for median in bp['medians']:
        median.set(color='black', linewidth=1.5)
        
    axes[1].set_ylabel("Sharpness Log10(Laplacian Var + 1)")
    axes[1].set_title("Sharpness Comparison (Boxplot)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sharpness_distribution.png"), bbox_inches='tight')
    plt.close()

def plot_color_distributions(df, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"live": "#2ca02c", "spoof": "#d62728"}
    
    # Hue mean
    for name, group in df.groupby("class"):
        axes[0].hist(group["hue_mean"], bins=50, alpha=0.5, label=name, color=colors[name], density=True)
    axes[0].set_xlabel("Mean Hue (0-179 in OpenCV)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Hue (Color Tone) Distribution")
    axes[0].legend()
    
    # Saturation mean
    for name, group in df.groupby("class"):
        axes[1].hist(group["sat_mean"], bins=50, alpha=0.5, label=name, color=colors[name], density=True)
    axes[1].set_xlabel("Mean Saturation (0-255)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Saturation (Color Purity) Distribution")
    axes[1].legend()
    
    # RGB comparison line charts or channel mean bar plots
    mean_rgb = df.groupby("class")[["r_mean", "g_mean", "b_mean"]].mean()
    x = np.arange(3)
    width = 0.35
    
    axes[2].bar(x - width/2, mean_rgb.loc["live"], width, label='Live', color=colors['live'], alpha=0.7)
    axes[2].bar(x + width/2, mean_rgb.loc["spoof"], width, label='Spoof', color=colors['spoof'], alpha=0.7)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(['Red Channel', 'Green Channel', 'Blue Channel'])
    axes[2].set_ylabel('Mean Intensity (0-255)')
    axes[2].set_title('Average RGB Color Channels')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "color_spaces.png"), bbox_inches='tight')
    plt.close()

def plot_fourier_analysis(fft_maps, output_dir):
    if not fft_maps["live"] or not fft_maps["spoof"]:
        return
        
    avg_live_fft = np.mean(fft_maps["live"], axis=0)
    avg_spoof_fft = np.mean(fft_maps["spoof"], axis=0)
    fft_diff = avg_live_fft - avg_spoof_fft
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Live Fourier Spectrum
    im0 = axes[0].imshow(avg_live_fft, cmap='viridis')
    axes[0].set_title("Avg Fourier Spectrum: LIVE")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].grid(False)
    
    # Spoof Fourier Spectrum
    im1 = axes[1].imshow(avg_spoof_fft, cmap='viridis')
    axes[1].set_title("Avg Fourier Spectrum: SPOOF")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].grid(False)
    
    # Difference Map
    im2 = axes[2].imshow(fft_diff, cmap='coolwarm')
    axes[2].set_title("Fourier Spectrum Difference (Live - Spoof)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].grid(False)
    
    plt.suptitle("Frequency Domain Analysis (Average Log Fourier Magnitudes)", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fourier_analysis.png"), bbox_inches='tight')
    plt.close()

def plot_dashboard(df, fft_maps, output_dir):
    """
    Creates a single consolidated dashboard summarizing all findings.
    """
    fig = plt.subplots(figsize=(20, 16))
    colors = {"live": "#2ca02c", "spoof": "#d62728"}
    
    # Grid specification
    grid_shape = (3, 3)
    
    # Row 0: Counts and Dimensions
    ax1 = plt.subplot2grid(grid_shape, (0, 0))
    counts = df.groupby(["split", "class"]).size().unstack(fill_value=0)
    counts.plot(kind='bar', color=[colors['live'], colors['spoof']], alpha=0.7, ax=ax1, edgecolor='black')
    ax1.set_title("Analysis Sample Distribution")
    ax1.set_ylabel("Count")
    ax1.set_xlabel("Dataset Split")
    ax1.tick_params(axis='x', rotation=0)
    
    ax2 = plt.subplot2grid(grid_shape, (0, 1))
    for name, group in df.groupby("class"):
        ax2.scatter(group["width"], group["height"], alpha=0.3, label=name, color=colors[name], s=15)
    ax2.set_xlabel("Width")
    ax2.set_ylabel("Height")
    ax2.set_title("Dimensions (W vs H)")
    ax2.legend()
    
    ax3 = plt.subplot2grid(grid_shape, (0, 2))
    for name, group in df.groupby("class"):
        ax3.hist(group["aspect_ratio"], bins=30, alpha=0.5, label=name, color=colors[name], density=True)
    ax3.set_xlabel("Aspect Ratio (W/H)")
    ax3.set_title("Aspect Ratio Distribution")
    ax3.legend()
    
    # Row 1: Brightness, Contrast, Sharpness
    ax4 = plt.subplot2grid(grid_shape, (1, 0))
    for name, group in df.groupby("class"):
        ax4.hist(group["brightness"], bins=40, alpha=0.5, label=name, color=colors[name], density=True)
    ax4.set_xlabel("Brightness (0-255)")
    ax4.set_title("Mean Brightness")
    ax4.legend()
    
    ax5 = plt.subplot2grid(grid_shape, (1, 1))
    for name, group in df.groupby("class"):
        ax5.hist(group["contrast"], bins=40, alpha=0.5, label=name, color=colors[name], density=True)
    ax5.set_xlabel("Contrast (Intensity Std)")
    ax5.set_title("RMS Contrast")
    ax5.legend()
    
    ax6 = plt.subplot2grid(grid_shape, (1, 2))
    for name, group in df.groupby("class"):
        ax6.hist(np.log10(group["sharpness"]+1), bins=40, alpha=0.5, label=name, color=colors[name], density=True)
    ax6.set_xlabel("Sharpness (Log Laplacian Var)")
    ax6.set_title("Sharpness / Blurriness")
    ax6.legend()
    
    # Row 2: Color Saturation and FFT
    ax7 = plt.subplot2grid(grid_shape, (2, 0))
    for name, group in df.groupby("class"):
        ax7.hist(group["sat_mean"], bins=40, alpha=0.5, label=name, color=colors[name], density=True)
    ax7.set_xlabel("Mean Saturation (0-255)")
    ax7.set_title("Color Saturation")
    ax7.legend()
    
    # Fourier Heatmaps in row 2, col 1 and 2
    if fft_maps["live"] and fft_maps["spoof"]:
        avg_live_fft = np.mean(fft_maps["live"], axis=0)
        avg_spoof_fft = np.mean(fft_maps["spoof"], axis=0)
        
        ax8 = plt.subplot2grid(grid_shape, (2, 1))
        im8 = ax8.imshow(avg_live_fft, cmap='viridis')
        ax8.set_title("Avg Fourier: LIVE")
        ax8.grid(False)
        plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)
        
        ax9 = plt.subplot2grid(grid_shape, (2, 2))
        im9 = ax9.imshow(avg_spoof_fft, cmap='viridis')
        ax9.set_title("Avg Fourier: SPOOF")
        ax9.grid(False)
        plt.colorbar(im9, ax=ax9, fraction=0.046, pad=0.04)
        
    plt.suptitle("Face Anti-Spoofing EDA Dashboard", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "eda_dashboard.png"), bbox_inches='tight')
    plt.close()

def write_markdown_report(output_dir, counts, df):
    # Calculate some key statistics to inject in the report
    live_df = df[df["class"] == "live"]
    spoof_df = df[df["class"] == "spoof"]
    
    with open(os.path.join(output_dir, "eda_report.md"), "w") as f:
        f.write("# Face Anti-Spoofing Image Dataset - Exploratory Data Analysis (EDA) Report\n\n")
        
        # 1. Dataset structure
        f.write("## 1. Dataset Dimensions and Split Balance\n\n")
        f.write("Total images found in the filesystem directories:\n\n")
        f.write("| Split | Live (0) | Spoof (1) | Total | Live % |\n")
        f.write("|---|---|---|---|---|\n")
        for split in ["train"]:
            l = counts.get(split, {}).get("live", 0)
            s = counts.get(split, {}).get("spoof", 0)
            tot = l + s
            pct = (l / tot * 100) if tot > 0 else 0
            f.write(f"| {split.capitalize()} | {l:,} | {s:,} | {tot:,} | {pct:.1f}% |\n")
        
        # 2. Sample size
        f.write(f"\n*Note: The remaining statistics below were computed on a randomized, balanced sample of **{len(df):,}** images to ensure fast computational throughput.*\n\n")
        
        # 3. Size and Aspect Ratio
        f.write("## 2. Image Spatial Characteristics\n\n")
        f.write("| Metric / Class | Live (Mean ± Std) | Spoof (Mean ± Std) | Live Median | Spoof Median |\n")
        f.write("|---|---|---|---|---|\n")
        f.write(f"| **Width (px)** | {live_df['width'].mean():.1f} ± {live_df['width'].std():.1f} | {spoof_df['width'].mean():.1f} ± {spoof_df['width'].std():.1f} | {live_df['width'].median():.1f} | {spoof_df['width'].median():.1f} |\n")
        f.write(f"| **Height (px)** | {live_df['height'].mean():.1f} ± {live_df['height'].std():.1f} | {spoof_df['height'].mean():.1f} ± {spoof_df['height'].std():.1f} | {live_df['height'].median():.1f} | {spoof_df['height'].median():.1f} |\n")
        f.write(f"| **Aspect Ratio (W/H)** | {live_df['aspect_ratio'].mean():.3f} ± {live_df['aspect_ratio'].std():.3f} | {spoof_df['aspect_ratio'].mean():.3f} ± {spoof_df['aspect_ratio'].std():.3f} | {live_df['aspect_ratio'].median():.3f} | {spoof_df['aspect_ratio'].median():.3f} |\n")
        f.write(f"| **Resolution (Area)** | {live_df['resolution'].mean():.1f} ± {live_df['resolution'].std():.1f} | {spoof_df['resolution'].mean():.1f} ± {spoof_df['resolution'].std():.1f} | {live_df['resolution'].median():.1f} | {spoof_df['resolution'].median():.1f} |\n")
        
        # 4. Color, Brightness & Sharpness
        f.write("\n## 3. Pixel Intensity & Texture Characteristics\n\n")
        f.write("| Metric / Class | Live (Mean ± Std) | Spoof (Mean ± Std) | Live Median | Spoof Median |\n")
        f.write("|---|---|---|---|---|\n")
        f.write(f"| **Brightness (Gray Mean)** | {live_df['brightness'].mean():.1f} ± {live_df['brightness'].std():.1f} | {spoof_df['brightness'].mean():.1f} ± {spoof_df['brightness'].std():.1f} | {live_df['brightness'].median():.1f} | {spoof_df['brightness'].median():.1f} |\n")
        f.write(f"| **Contrast (Gray Std)** | {live_df['contrast'].mean():.1f} ± {live_df['contrast'].std():.1f} | {spoof_df['contrast'].mean():.1f} ± {spoof_df['contrast'].std():.1f} | {live_df['contrast'].median():.1f} | {spoof_df['contrast'].median():.1f} |\n")
        f.write(f"| **Sharpness (Laplacian Var)** | {live_df['sharpness'].mean():.1f} ± {live_df['sharpness'].std():.1f} | {spoof_df['sharpness'].mean():.1f} ± {spoof_df['sharpness'].std():.1f} | {live_df['sharpness'].median():.1f} | {spoof_df['sharpness'].median():.1f} |\n")
        f.write(f"| **Saturation (HSV S-Mean)** | {live_df['sat_mean'].mean():.1f} ± {live_df['sat_mean'].std():.1f} | {spoof_df['sat_mean'].mean():.1f} ± {spoof_df['sat_mean'].std():.1f} | {live_df['sat_mean'].median():.1f} | {spoof_df['sat_mean'].median():.1f} |\n")
        
        # 5. Domain specific findings
        f.write("\n## 4. Key Observations & Modeling Guidance\n\n")
        
        # Check brightness difference
        b_diff = live_df['brightness'].mean() - spoof_df['brightness'].mean()
        if abs(b_diff) > 10:
            b_obs = f"Spoof images are on average {'darker' if b_diff > 0 else 'brighter'} than live images by {abs(b_diff):.1f} intensity levels. Consider using Random Brightness/Contrast augmentations during training to prevent the model from learning simple illumination biases."
        else:
            b_obs = "Mean brightness between Live and Spoof images is comparable (less than 10 levels of difference)."
            
        # Check sharpness difference
        s_ratio = live_df['sharpness'].mean() / (spoof_df['sharpness'].mean() + 1e-5)
        if s_ratio > 1.2:
            s_obs = f"Live images are significantly sharper than spoof images (average Laplacian variance ratio: {s_ratio:.2f}x). Spoof images suffer from blurriness due to re-capture effects (printing/screen display). Do not use excessive blur augmentations on spoof images, and consider adding RandomBlur/GaussianNoise to live images to make the model robust."
        elif s_ratio < 0.8:
            s_obs = f"Spoof images are significantly sharper than live images (average Laplacian variance ratio: {1.0/s_ratio:.2f}x)."
        else:
            s_obs = "Image sharpness is relatively similar between live and spoof categories."
            
        # Check resolution
        res_live = live_df['resolution'].median()
        res_spoof = spoof_df['resolution'].median()
        if abs(res_live - res_spoof) / max(res_live, res_spoof) > 0.15:
            r_obs = f"There is a visible resolution mismatch (Live median: {res_live:,.0f}px, Spoof median: {res_spoof:,.0f}px). Preprocessing pipelines MUST resize images consistently to avoid resolution-based shortcuts. Using SquarePad + Resize is highly recommended."
        else:
            r_obs = "Resolution distributions are highly matched, ensuring minimal bias from resolution shortcuts."
            
        f.write(f"- **Illumination**: {b_obs}\n")
        f.write(f"- **Sharpness/Focus**: {s_obs}\n")
        f.write(f"- **Spatial Resolution**: {r_obs}\n")
        f.write("- **Fourier/Frequency Domain**: Frequency analysis shows the average energy distribution of live vs spoof faces. Spoof faces printed on paper or displayed on screens often exhibit regular patterns (moiré, print dots) or high-frequency damping, which shows up as differences in the outer parts of the Fourier spectrum. Using Fourier features (`use_fourier=True` in dataset) can help capture these clues.\n")

if __name__ == "__main__":
    main()
