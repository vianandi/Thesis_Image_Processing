import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
from torch.utils.data import DataLoader
from utils.dataset import LOLDataset
from utils.baselines import apply_clahe, apply_gamma, apply_retinex
from utils.matrics import psnr, calc_ssim, calc_niqe, calc_loe

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("PyTorch version:", torch.__version__)
print("CUDA available? ", torch.cuda.is_available())
print("CUDA version (compiled):", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
print("GPU device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

def evaluate_baseline(dataset_dir, output_csv="baseline_results.csv", max_images=20):
    ds = LOLDataset(dataset_dir, split="val")
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    
    print(f"Total images available: {len(ds)}")
    print(f"Processing first {min(max_images, len(ds))} images")
    
    results = []
    processed_count = 0
    
    for idx, (low, high, name) in enumerate(loader):
        if processed_count >= max_images:
            break
            
        print(f"Processing image {processed_count+1}/{min(max_images, len(loader))}: {name}")
        low, high = low[0].to(device), high[0].to(device)

        for method_name, func in {
            "clahe": apply_clahe,
            "gamma": lambda x: apply_gamma(x, gamma=2.2),
            "retinex": apply_retinex
        }.items():
            print(f"  Applying {method_name}...")
            enh = func(low.cpu()).to(device)

            print(f"  Calculating metrics for {method_name}...")
            scores = {
                "image": name,
                "method": method_name,
                "psnr": psnr(enh.cpu(), high.cpu()).item(),
                "ssim": calc_ssim(enh.cpu(), high.cpu()),
                "niqe": calc_niqe(enh.cpu()),
                "loe": calc_loe(low.cpu(), enh.cpu())
            }
            results.append(scores)
            print(f"  {method_name} completed!")
        
        processed_count += 1

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")
    return df

if __name__ == "__main__":
    # Proses hanya 5 gambar pertama untuk testing cepat
    evaluate_baseline("dataset/our485", max_images=20)
