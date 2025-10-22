# 6_visualize.py - Comprehensive visualization and analysis for Lab 8
"""
Creates publication-quality visualizations comparing Doc2Vec and Word2Vec binning methods.
Outputs all plots and analysis to ../comparison_results/
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import joblib

# Configuration
RESULTS_DIR = Path('../comparison_results')
PLOTS_DIR = RESULTS_DIR / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("COMPREHENSIVE VISUALIZATION & ANALYSIS - LAB 8")
print("=" * 80)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_embeddings(method, dim):
    """Load document embeddings for a given method and dimension."""
    if method == 'Doc2Vec':
        path = f'../results/doc2vec_embeddings/docvecs_vs{dim}.pkl'
    else:  # Word2Vec_bins
        path = f'../results/word2vec_bins/K{dim}/docvecs.pkl'
    
    try:
        X = np.array(joblib.load(path))
        return normalize(X, norm='l2')
    except FileNotFoundError:
        print(f"Not found: {path}")
        return None


def load_labels(method, dim, k):
    """Load cluster labels for a given configuration."""
    if method == 'Doc2Vec':
        path = f'../results/clustering/doc2vec_vs{dim}/k{k}/labels.csv'
    else:
        path = f'../results/clustering/wordbins_K{dim}/k{k}/labels.csv'
    
    try:
        return pd.read_csv(path)['label'].values
    except FileNotFoundError:
        print(f"Labels not found: {path}")
        return None


# ============================================================================
# VISUALIZATION 1: PCA Scatter Plots (Best Configurations)
# ============================================================================

def create_pca_scatter(X, labels, method, dim, k, metrics, save_path):
    """Create PCA scatter plot with cluster colors."""
    # Reduce to 2D
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        X_2d[:, 0], X_2d[:, 1], 
        c=labels, 
        cmap='tab10', 
        s=10, 
        alpha=0.6
    )
    
    # Title with metrics
    title = f"{method} (dim={dim}, K={k}) | Sil={metrics['silhouette']:.3f}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('PCA 1', fontsize=12)
    ax.set_ylabel('PCA 2', fontsize=12)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Cluster ID')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path.name}")


# ============================================================================
# VISUALIZATION 2: Metric Comparison Bar Charts
# ============================================================================

def create_metric_comparison_bars(df, save_dir):
    """Create bar charts comparing metrics across methods and dimensions."""
    metrics = ['silhouette', 'davies_bouldin', 'calinski_harabasz']
    titles = ['Silhouette Score (Higher=Better)', 
              'Davies-Bouldin Index (Lower=Better)', 
              'Calinski-Harabasz Score (Higher=Better)']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, metric, title in zip(axes, metrics, titles):
        # Group by method and dimension
        pivot = df.pivot_table(
            values=metric, 
            index='dim', 
            columns='method', 
            aggfunc='max'  # Best score for each dim
        )
        
        pivot.plot(kind='bar', ax=ax, width=0.7)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Embedding Dimension', fontsize=11)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.legend(title='Method', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        # Rotate x-axis labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    plt.tight_layout()
    save_path = save_dir / 'metric_comparison_bars.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path.name}")


# ============================================================================
# VISUALIZATION 3: Heatmaps (Dimension × K Clusters)
# ============================================================================

def create_heatmap(df, method, metric, save_dir):
    """Create heatmap showing metric values across dimension and k."""
    # Filter for one method
    method_df = df[df['method'] == method]
    
    if len(method_df) == 0:
        return
    
    # Pivot: rows=dimension, columns=n_clusters
    pivot = method_df.pivot_table(
        values=metric,
        index='dim',
        columns='n_clusters',
        aggfunc='mean'
    )
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Reverse colormap for davies_bouldin (lower is better)
    if metric == 'davies_bouldin':
        cmap = 'RdYlGn'  # Red=bad, Green=good
    else:
        cmap = 'RdYlGn_r'  # Red=good, Green=bad
    
    sns.heatmap(
        pivot, 
        annot=True, 
        fmt='.3f', 
        cmap=cmap,
        cbar_kws={'label': metric.replace('_', ' ').title()},
        ax=ax
    )
    
    ax.set_title(f'{method}: {metric.replace("_", " ").title()} Heatmap', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Embedding Dimension', fontsize=12)
    
    plt.tight_layout()
    save_path = save_dir / f'heatmap_{method}_{metric}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path.name}")


# ============================================================================
# VISUALIZATION 4: Side-by-Side Best Configurations
# ============================================================================

def create_side_by_side_comparison(df, save_dir):
    """Create side-by-side PCA plots for best configurations of each method."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, method in enumerate(['Doc2Vec', 'Word2Vec_bins']):
        method_df = df[df['method'] == method]
        
        if len(method_df) == 0:
            continue
        
        # Find best configuration (highest silhouette)
        best_row = method_df.loc[method_df['silhouette'].idxmax()]
        dim = int(best_row['dim'])
        k = int(best_row['n_clusters'])
        
        # Load data
        X = load_embeddings(method, dim)
        labels = load_labels(method, dim, k)
        
        if X is None or labels is None:
            continue
        
        # PCA
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X)
        
        # Plot
        ax = axes[idx]
        scatter = ax.scatter(
            X_2d[:, 0], X_2d[:, 1],
            c=labels,
            cmap='tab10',
            s=15,
            alpha=0.6
        )
        
        title = f"{method}\nBest: dim={dim}, k={k} | Sil={best_row['silhouette']:.3f}"
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('PCA 1', fontsize=11)
        ax.set_ylabel('PCA 2', fontsize=11)
        plt.colorbar(scatter, ax=ax, label='Cluster')
    
    plt.suptitle('Best Configuration Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = save_dir / 'side_by_side_best.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path.name}")


# ============================================================================
# VISUALIZATION 5: Method Performance Summary
# ============================================================================

def create_performance_summary(df, save_dir):
    """Create a summary visualization showing which method performs better."""
    # Get best score for each method across all configurations
    summary = df.groupby('method').agg({
        'silhouette': 'max',
        'davies_bouldin': 'min',  # Lower is better
        'calinski_harabasz': 'max'
    }).reset_index()
    
    # Normalize scores to 0-100 scale for comparison
    # Silhouette: already 0-1, multiply by 100
    # Davies-Bouldin: invert (1/x) and scale
    # Calinski-Harabasz: normalize to max
    
    summary['silhouette_norm'] = summary['silhouette'] * 100
    summary['davies_bouldin_norm'] = (1 / summary['davies_bouldin']) * 100
    summary['calinski_harabasz_norm'] = (
        summary['calinski_harabasz'] / summary['calinski_harabasz'].max() * 100
    )
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(summary))
    width = 0.25
    
    bars1 = ax.bar(x - width, summary['silhouette_norm'], width, 
                   label='Silhouette', color='#2ecc71')
    bars2 = ax.bar(x, summary['davies_bouldin_norm'], width,
                   label='Davies-Bouldin (inverted)', color='#3498db')
    bars3 = ax.bar(x + width, summary['calinski_harabasz_norm'], width,
                   label='Calinski-Harabasz', color='#e74c3c')
    
    ax.set_ylabel('Normalized Score (0-100, Higher=Better)', fontsize=12)
    ax.set_title('Method Performance Summary (Best Scores)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(summary['method'], fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    save_path = save_dir / 'performance_summary.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path.name}")


# ============================================================================
# VISUALIZATION 6: Dimensionality Effect Analysis
# ============================================================================

def create_dimensionality_effect(df, save_dir):
    """Show how dimensionality affects performance for each method."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = ['silhouette', 'davies_bouldin', 'calinski_harabasz']
    titles = ['Silhouette vs Dimension', 'Davies-Bouldin vs Dimension', 
              'Calinski-Harabasz vs Dimension']
    
    for ax, metric, title in zip(axes, metrics, titles):
        for method in df['method'].unique():
            method_df = df[df['method'] == method]
            
            # Get best score for each dimension
            best_per_dim = method_df.groupby('dim')[metric].agg(
                'min' if metric == 'davies_bouldin' else 'max'
            ).reset_index()
            
            ax.plot(best_per_dim['dim'], best_per_dim[metric], 
                   marker='o', linewidth=2, markersize=8, label=method)
        
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Embedding Dimension', fontsize=11)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = save_dir/'dimensionality_effect.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path.name}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all visualizations."""
    
    # Load evaluation results
    eval_path = Path('../results/clustering/evaluation_summary.csv')
    
    if not eval_path.exists():
        print(f"\nERROR: {eval_path} not found!")
        print("Run 4_cluster_eval_fixed.py first to generate results.\n")
        return
    
    df = pd.read_csv(eval_path)
    print(f"\nLoaded evaluation results: {len(df)} configurations")
    print(f"  Methods: {df['method'].unique()}")
    print(f"  Dimensions: {sorted(df['dim'].unique())}")
    print(f"  Cluster counts: {sorted(df['n_clusters'].unique())}")
    
    # ========================================================================
    # 1. PCA Scatter Plots for Best Configurations
    # ========================================================================
    print("\n[1/6] Creating PCA scatter plots for best configurations...")
    
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        
        # Best overall
        best_row = method_df.loc[method_df['silhouette'].idxmax()]
        dim = int(best_row['dim'])
        k = int(best_row['n_clusters'])
        
        X = load_embeddings(method, dim)
        labels = load_labels(method, dim, k)
        
        if X is not None and labels is not None:
            metrics = {
                'silhouette': best_row['silhouette'],
                'davies_bouldin': best_row['davies_bouldin'],
                'calinski_harabasz': best_row['calinski_harabasz']
            }
            save_path = PLOTS_DIR / f'{method}_best_dim{dim}_k{k}.png'
            create_pca_scatter(X, labels, method, dim, k, metrics, save_path)
    
    # ========================================================================
    # 2. Metric Comparison Bar Charts
    # ========================================================================
    print("\n[2/6] Creating metric comparison bar charts...")
    create_metric_comparison_bars(df, PLOTS_DIR)
    
    # ========================================================================
    # 3. Heatmaps for Each Method
    # ========================================================================
    print("\n[3/6] Creating heatmaps...")
    for method in df['method'].unique():
        for metric in ['silhouette', 'davies_bouldin', 'calinski_harabasz']:
            create_heatmap(df, method, metric, PLOTS_DIR)
    
    # ========================================================================
    # 4. Side-by-Side Best Comparison
    # ========================================================================
    print("\n[4/6] Creating side-by-side comparison...")
    create_side_by_side_comparison(df, PLOTS_DIR)
    
    # ========================================================================
    # 5. Performance Summary
    # ========================================================================
    print("\n[5/6] Creating performance summary...")
    create_performance_summary(df, PLOTS_DIR)
    
    # ========================================================================
    # 6. Dimensionality Effect Analysis
    # ========================================================================
    print("\n[6/6] Creating dimensionality effect analysis...")
    create_dimensionality_effect(df, PLOTS_DIR)
    
    # ========================================================================
    # Save Enhanced CSV with Rankings
    # ========================================================================
    print("\n[BONUS] Creating enhanced results CSV...")
    
    # Add rankings
    df['silhouette_rank'] = df.groupby('method')['silhouette'].rank(ascending=False)
    df['overall_rank'] = df.groupby('method').apply(
        lambda x: x['silhouette'].rank(ascending=False)
    ).values
    
    # Save
    enhanced_path = RESULTS_DIR / 'evaluation_summary_with_rankings.csv'
    df.to_csv(enhanced_path, index=False)
    print(f"Saved: {enhanced_path}")
    
    # Print best configurations
    print("\n" + "=" * 80)
    print("BEST CONFIGURATIONS")
    print("=" * 80)
    
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        best = method_df.loc[method_df['silhouette'].idxmax()]
        
        print(f"\n{method}:")
        print(f"  Dimension: {int(best['dim'])}")
        print(f"  Clusters (k): {int(best['n_clusters'])}")
        print(f"  Silhouette: {best['silhouette']:.4f}")
        print(f"  Davies-Bouldin: {best['davies_bouldin']:.4f}")
        print(f"  Calinski-Harabasz: {best['calinski_harabasz']:.1f}")
    
    print("\n" + "=" * 80)
    print("ALL VISUALIZATIONS COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to:")
    print(f"CSV: {enhanced_path}")
    print(f"Plots: {PLOTS_DIR}")
    print(f"\nTotal plots created: {len(list(PLOTS_DIR.glob('*.png')))}")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
