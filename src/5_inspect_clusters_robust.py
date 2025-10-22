# 5_inspect_clusters_robust.py - Robust cluster inspection with error handling
"""
This script examines actual documents in each cluster to assess semantic coherence.
Handles missing/malformed data gracefully.
"""
import pandas as pd
import numpy as np
import joblib
import os

# Load original preprocessed data
df = pd.read_pickle('../data/reddit_posts_prepped.pkl')

def inspect_clustering(method, dim, k, n_samples=5):
    """
    Inspect a specific clustering result by showing sample documents from each cluster.
    
    Args:
        method: 'doc2vec' or 'wordbins'
        dim: dimension/K value (50, 100, or 300)
        k: number of clusters
        n_samples: number of sample documents to show per cluster
    """
    # Load cluster labels
    if method == 'doc2vec':
        labels_path = f'../results/clustering/doc2vec_vs{dim}/k{k}/labels.csv'
    else:  # wordbins
        labels_path = f'../results/clustering/wordbins_K{dim}/k{k}/labels.csv'
    
    if not os.path.exists(labels_path):
        print(f"  [SKIP] Labels not found: {labels_path}")
        return None
    
    labels = pd.read_csv(labels_path)['label'].values
    df_copy = df.copy()
    df_copy['cluster'] = labels
    
    print(f"\n{'=' * 80}")
    print(f"CLUSTER INSPECTION: {method.upper()} | dim={dim} | k={k}")
    print(f"{'=' * 80}")
    
    cluster_info = []
    
    for cluster_id in range(k):
        cluster_docs = df_copy[df_copy['cluster'] == cluster_id]
        n_docs = len(cluster_docs)
        
        print(f"\n--- Cluster {cluster_id} (n={n_docs} documents) ---")
        
        if n_docs == 0:
            print("  [EMPTY CLUSTER]")
            cluster_info.append({
                'cluster_id': cluster_id,
                'size': 0,
                'quality': 'empty'
            })
            continue
        
        # Show sample documents
        samples = cluster_docs.head(n_samples)
        for idx, (row_idx, row) in enumerate(samples.iterrows(), 1):
            # Safely reconstruct text from tokens
            try:
                tokens = row['tokens']
                
                # Handle different types of bad data
                if tokens is None:
                    text_preview = "[NULL TOKENS]"
                elif isinstance(tokens, float) and np.isnan(tokens):
                    text_preview = "[NaN TOKENS]"
                elif not isinstance(tokens, list):
                    text_preview = f"[INVALID TYPE: {type(tokens).__name__}]"
                elif len(tokens) == 0:
                    text_preview = "[EMPTY TOKEN LIST]"
                else:
                    # Normal case: join tokens
                    text_preview = ' '.join(str(t) for t in tokens[:20])
                    if len(tokens) > 20:
                        text_preview += '...'
                    
                    # Flag if it looks like garbage
                    if all(str(t) in ['none', 'nan', 'null', ''] for t in tokens[:5]):
                        text_preview = f"[GARBAGE] {text_preview}"
                
                print(f"  [{idx}] {text_preview}")
                
            except Exception as e:
                print(f"  [{idx}] [ERROR: {e}]")
        
        cluster_info.append({
            'cluster_id': cluster_id,
            'size': n_docs,
            'quality': 'needs_manual_assessment'
        })
    
    return cluster_info


def main():
    """
    Systematically inspect key clustering configurations.
    """
    
    print("=" * 80)
    print("QUALITATIVE CLUSTER ANALYSIS")
    print("=" * 80)
    print("\nInspecting key configurations to assess semantic coherence...")
    
    # Key configurations to inspect
    configs = [
        # Doc2Vec configurations
        ('doc2vec', 50, 5),
        ('doc2vec', 100, 5),
        ('doc2vec', 300, 4),
        
        # Word2Vec binning configurations  
        ('wordbins', 50, 5),
        ('wordbins', 100, 4),
        ('wordbins', 300, 5),
    ]
    
    for method, dim, k in configs:
        inspect_clustering(method, dim, k, n_samples=5)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review the sample documents above")
    print("2. Assess whether documents in each cluster share semantic themes")
    print("3. Compare coherence between Doc2Vec and Word2Vec_bins methods")
    print("4. If you see many [GARBAGE] or [NULL] tags, re-run data preparation")
    print("\n")


if __name__ == '__main__':
    main()
