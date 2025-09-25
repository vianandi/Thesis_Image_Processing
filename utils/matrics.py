import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
import lpips
from PIL import Image

_lpips_model = None

def get_lpips_model():
    """Get or initialize LPIPS model (singleton pattern)"""
    global _lpips_model
    if _lpips_model is None:
        print("Initializing LPIPS model...")
        _lpips_model = lpips.LPIPS(net='alex')
        print("LPIPS model loaded successfully!")
    return _lpips_model

# Metric calculation functions
def psnr(img1, img2):
    """Calculate PSNR between two images"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def calc_ssim(img1, img2):
    """Calculate SSIM between two images"""
    img1_np = img1.permute(1, 2, 0).numpy()
    img2_np = img2.permute(1, 2, 0).numpy()
    
    # Get image dimensions
    h, w, c = img1_np.shape
    
    # Check if images are large enough for default window size (7x7)
    min_dim = min(h, w)
    if min_dim < 7:
        # Use smaller window size for small images
        win_size = min_dim if min_dim % 2 == 1 else min_dim - 1
        if win_size < 3:
            win_size = 3
    else:
        win_size = 7
    
    # Use channel_axis instead of multichannel (which is deprecated)
    return ssim(img1_np, img2_np, channel_axis=2, data_range=1.0, win_size=win_size)


def calc_lpips(img1, img2):
    """Calculate LPIPS between two images"""
    loss_fn = lpips.LPIPS(net='alex')
    # Ensure images are in the correct format [-1, 1]
    img1_norm = img1 * 2.0 - 1.0
    img2_norm = img2 * 2.0 - 1.0
    return loss_fn(img1_norm, img2_norm).item()

def calc_niqe(img):
    """Calculate NIQE score for an image"""
    # Simple implementation - you might want to use a more sophisticated NIQE implementation
    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    gray = np.dot(img_np[...,:3], [0.2989, 0.5870, 0.1140])
    
    # Calculate local variance as a simple quality measure
    kernel = np.ones((3,3)) / 9
    local_mean = np.convolve(gray.flatten(), kernel.flatten(), mode='same').reshape(gray.shape)
    local_var = np.convolve((gray - local_mean).flatten()**2, kernel.flatten(), mode='same').reshape(gray.shape)
    
    return np.mean(local_var)

def calc_loe(low_img, enhanced_img):
    """Calculate Loss of Exposure (LOE) metric - optimized version"""
    low_np = low_img.permute(1, 2, 0).numpy()
    enh_np = enhanced_img.permute(1, 2, 0).numpy()
    
    # Convert to grayscale
    low_gray = np.mean(low_np, axis=2)
    enh_gray = np.mean(enh_np, axis=2)
    
    # Downsample for faster computation
    if low_gray.shape[0] > 64 or low_gray.shape[1] > 64:
        from skimage.transform import resize
        target_size = (64, 64)
        low_gray = resize(low_gray, target_size, anti_aliasing=True)
        enh_gray = resize(enh_gray, target_size, anti_aliasing=True)
    
    h, w = low_gray.shape
    
    # Vectorized computation instead of nested loops
    low_flat = low_gray.flatten()
    enh_flat = enh_gray.flatten()
    
    # Sample random pairs instead of all pairs to speed up
    n_samples = min(1000, len(low_flat))
    indices = np.random.choice(len(low_flat), n_samples, replace=False)
    
    loe_count = 0
    for i in indices:
        for j in indices:
            if (low_flat[i] < low_flat[j]) != (enh_flat[i] < enh_flat[j]):
                loe_count += 1
    
    return loe_count / (n_samples * n_samples)

# Visualization code (existing code)
def visualize_results():
    # Load the results from the CSV file
    results_df = pd.read_csv("baseline_results.csv")

    # Calculate the average metrics for each method
    average_metrics = results_df.groupby("method")[["psnr", "ssim", "lpips", "niqe", "loe"]].mean()

    # Display the average metrics
    print("Average Metrics per Method:")
    print(average_metrics)
     
    # Create directory for visualizations
    output_viz_dir = "output_baseline_method"
    os.makedirs(output_viz_dir, exist_ok=True)

    # Visualize and save the average metrics
    metrics_to_visualize = ["psnr", "ssim", "lpips", "niqe", "loe"]

    for metric in metrics_to_visualize:
        plt.figure(figsize=(8, 5))
        sns.barplot(x=average_metrics.index, y=average_metrics[metric])
        plt.title(f"Average {metric.upper()} Comparison of Baselines")
        plt.ylabel(f"Average {metric.upper()}")
        plt.xlabel("Method")

        # Save the plot
        save_path = os.path.join(output_viz_dir, f"average_{metric.lower()}_comparison.png")
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
        plt.close() # Close the plot to free memory

if __name__ == "__main__":
    visualize_results()