import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_baseline_visualizations(csv_file="baseline_results.csv"):
    """Create comprehensive visualizations for baseline evaluation results"""
    
    # Load results
    df = pd.read_csv(csv_file)
    print(f"Loaded results for {len(df)} evaluations")
    print(f"Methods: {df['method'].unique()}")
    print(f"Images: {len(df['image'].unique())}")
    
    # Create output directory
    output_dir = "output_baseline_method"
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate average metrics per method
    avg_metrics = df.groupby('method')[['psnr', 'ssim', 'niqe', 'loe']].mean()
    std_metrics = df.groupby('method')[['psnr', 'ssim', 'niqe', 'loe']].std()
    
    print("\nAverage metrics per method:")
    print(avg_metrics)
    
    # Set style for better plots
    plt.style.use('seaborn-v0_8')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # 1. Bar plot for each metric
    metrics = ['psnr', 'ssim', 'niqe', 'loe']
    metric_names = ['PSNR (↑)', 'SSIM (↑)', 'NIQE (↓)', 'LOE (↓)']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i]
        bars = ax.bar(avg_metrics.index, avg_metrics[metric], 
                     yerr=std_metrics[metric], capsize=5, 
                     color=colors[i], alpha=0.7, edgecolor='black')
        
        ax.set_title(f'{name} Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel(name, fontsize=12)
        ax.set_xlabel('Method', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, avg_metrics[metric]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'baseline_metrics_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Radar chart for overall comparison
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Normalize metrics for radar chart (0-1 scale)
    norm_metrics = avg_metrics.copy()
    # For PSNR and SSIM (higher is better), normalize to 0-1
    norm_metrics['psnr'] = (norm_metrics['psnr'] - norm_metrics['psnr'].min()) / (norm_metrics['psnr'].max() - norm_metrics['psnr'].min())
    norm_metrics['ssim'] = (norm_metrics['ssim'] - norm_metrics['ssim'].min()) / (norm_metrics['ssim'].max() - norm_metrics['ssim'].min())
    # For NIQE and LOE (lower is better), invert and normalize
    norm_metrics['niqe'] = 1 - (norm_metrics['niqe'] - norm_metrics['niqe'].min()) / (norm_metrics['niqe'].max() - norm_metrics['niqe'].min())
    norm_metrics['loe'] = 1 - (norm_metrics['loe'] - norm_metrics['loe'].min()) / (norm_metrics['loe'].max() - norm_metrics['loe'].min())
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for i, method in enumerate(norm_metrics.index):
        values = norm_metrics.loc[method].tolist()
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1)
    ax.set_title('Overall Performance Comparison\n(Normalized Metrics)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.savefig(os.path.join(output_dir, 'baseline_radar_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Box plots for metric distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i]
        box_plot = ax.boxplot([df[df['method'] == method][metric] for method in df['method'].unique()],
                             labels=df['method'].unique(), patch_artist=True)
        
        # Color the boxes
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title(f'{name} Distribution', fontsize=14, fontweight='bold')
        ax.set_ylabel(name, fontsize=12)
        ax.set_xlabel('Method', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'baseline_distribution_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Heatmap of all results
    plt.figure(figsize=(12, 8))
    
    # Pivot data for heatmap
    heatmap_data = df.set_index(['image', 'method'])[metrics].unstack('method')
    
    # Create subplots for each metric
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i]
        data = heatmap_data[metric]
        
        sns.heatmap(data, annot=True, fmt='.3f', cmap='viridis', 
                   ax=ax, cbar_kws={'label': name})
        ax.set_title(f'{name} Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Image', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'baseline_heatmap_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Summary table
    summary_table = pd.DataFrame({
        'Method': avg_metrics.index,
        'PSNR (↑)': [f"{avg_metrics.loc[m, 'psnr']:.3f} ± {std_metrics.loc[m, 'psnr']:.3f}" for m in avg_metrics.index],
        'SSIM (↑)': [f"{avg_metrics.loc[m, 'ssim']:.3f} ± {std_metrics.loc[m, 'ssim']:.3f}" for m in avg_metrics.index],
        'NIQE (↓)': [f"{avg_metrics.loc[m, 'niqe']:.3f} ± {std_metrics.loc[m, 'niqe']:.3f}" for m in avg_metrics.index],
        'LOE (↓)': [f"{avg_metrics.loc[m, 'loe']:.3f} ± {std_metrics.loc[m, 'loe']:.3f}" for m in avg_metrics.index]
    })
    
    print("\nSummary Table (Mean ± Std):")
    print(summary_table.to_string(index=False))
    
    # Save summary table
    summary_table.to_csv(os.path.join(output_dir, 'baseline_summary_table.csv'), index=False)
    
    # 6. Ranking analysis
    print("\nRanking Analysis:")
    print("="*50)
    
    # Rank methods for each metric
    rankings = {}
    for metric in metrics:
        if metric in ['psnr', 'ssim']:  # Higher is better
            rankings[metric] = avg_metrics[metric].rank(ascending=False)
        else:  # Lower is better
            rankings[metric] = avg_metrics[metric].rank(ascending=True)
    
    ranking_df = pd.DataFrame(rankings)
    avg_rank = ranking_df.mean(axis=1).sort_values()
    
    print("Average Ranking (1=best):")
    for method, rank in avg_rank.items():
        print(f"{method:>10}: {rank:.2f}")
    
    print(f"\nAll visualizations saved to: {output_dir}/")
    
    return avg_metrics, summary_table

if __name__ == "__main__":
    create_baseline_visualizations()
