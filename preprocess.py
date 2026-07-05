import os
import cv2
import argparse
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="Stage 1: Preprocess dataset by padding small images to at least 256x256.")
    parser.add_argument("--data-dir", type=str, default="./data/celeba-spoof-cropped-sampled/dataset",
                        help="Path to the dataset directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only scan and report what would be changed without writing files")
    return parser.parse_args()

def preprocess_images():
    args = get_args()
    data_dir = args.data_dir
    
    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory {data_dir} does not exist.")
        return
        
    print(f"Scanning dataset under: {data_dir}")
    if args.dry_run:
        print("[DRY RUN MODE] No changes will be written to disk.")
        

    image_extensions = (".png", ".jpg", ".jpeg", ".bmp")
    all_image_paths = []
    
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(image_extensions):
                all_image_paths.append(os.path.join(root, file))
                
    total_images = len(all_image_paths)
    print(f"Found {total_images} images to process.")
    
    padded_count = 0
    skipped_count = 0
    error_count = 0
    
    for path in tqdm(all_image_paths, desc="Processing images"):
        img = cv2.imread(path)
        if img is None:
            print(f"\nWarning: Could not read image {path}")
            error_count += 1
            continue
            
        h, w = img.shape[:2]
        
        # Check if padding is needed (if either dimension is less than 256)
        if h < 256 or w < 256:
            target_h = max(h, 256)
            target_w = max(w, 256)
            
            pad_h = target_h - h
            pad_w = target_w - w
            
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            
            padded_count += 1
            
            if not args.dry_run:
                # Apply padding using boundary reflection (Reflect Padding)
                padded_img = cv2.copyMakeBorder(
                    img, 
                    pad_top, 
                    pad_bottom, 
                    pad_left, 
                    pad_right, 
                    borderType=cv2.BORDER_REFLECT_101
                )
                # Overwrite original image
                cv2.imwrite(path, padded_img)
        else:
            skipped_count += 1
            
    print("\n" + "="*40)
    print("PREPROCESSING SUMMARY")
    print("="*40)
    print(f"Total scanned: {total_images}")
    print(f"Padded & overwritten: {padded_count}")
    print(f"Skipped (already >= 256x256): {skipped_count}")
    print(f"Errors loading: {error_count}")
    print("="*40)

if __name__ == "__main__":
    preprocess_images()
