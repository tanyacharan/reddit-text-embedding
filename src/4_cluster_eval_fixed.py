# 4_cluster_eval.py - Complete clustering evaluation for both methods
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import normalize

# Configuration
vector_sizes = [50, 100, 300]
cluster_choices = [4, 5, 8, 10, 15]

# Store all results for comparison
all_results = []

print("=" * 70)
print("CLUSTERING EVALUATION - BOTH METHODS")
print("=" * 70)

# ===== DOC2VEC EVALUATION =====
print("\n--- DOC2VEC CLUSTERING ---")
for vs in vector_sizes:
    docvecs_path = f'../results/doc2vec_embeddings/docvecs_vs{vs}.pkl'
    try:
        X = np.array(joblib.load(docvecs_path))
    except FileNotFoundError:
        print(f"  [SKIP] Doc2Vec vectors for vs={vs} not found")
        continue
    
    Xn = normalize(X, norm='l2')
    
    for k in cluster_choices:
        # Cluster
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(Xn)
        
        # Compute metrics
        sil = silhouette_score(Xn, labels, metric='cosine')
        db = davies_bouldin_score(Xn, labels)
        ch = calinski_harabasz_score(Xn, labels)
        
        print(f"  doc2vec vs={vs:3d} k={k:2d} | Sil={sil:.4f} DB={db:.4f} CH={ch:.1f}")
        
        # Save labels
        output_dir = f'../results/clustering/doc2vec_vs{vs}/k{k}'
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame({'label': labels}).to_csv(f'{output_dir}/labels.csv', index=False)
        
        # Store results
        all_results.append({
            'method': 'Doc2Vec',
            'dim': vs,
            'n_clusters': k,
            'silhouette': sil,
            'davies_bouldin': db,
            'calinski_harabasz': ch
        })

# ===== WORD2VEC BINNING EVALUATION =====
print("\n--- WORD2VEC BINNING CLUSTERING ---")
for K in vector_sizes:
    docvecs_path = f'../results/word2vec_bins/K{K}/docvecs.pkl'
    try:
        X = np.array(joblib.load(docvecs_path))
    except FileNotFoundError:
        print(f"  [SKIP] Word2Vec bins for K={K} not found")
        continue
    
    Xn = normalize(X, norm='l2')
    
    for k in cluster_choices:
        # Cluster
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(Xn)
        
        # Compute metrics
        sil = silhouette_score(Xn, labels, metric='cosine')
        db = davies_bouldin_score(Xn, labels)
        ch = calinski_harabasz_score(Xn, labels)
        
        print(f"  wordbins K={K:3d} k={k:2d} | Sil={sil:.4f} DB={db:.4f} CH={ch:.1f}")
        
        # Save labels
        output_dir = f'../results/clustering/wordbins_K{K}/k{k}'
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame({'label': labels}).to_csv(f'{output_dir}/labels.csv', index=False)
        
        # Store results
        all_results.append({
            'method': 'Word2Vec_bins',
            'dim': K,
            'n_clusters': k,
            'silhouette': sil,
            'davies_bouldin': db,
            'calinski_harabasz': ch
        })

# ===== SAVE SUMMARY =====
results_df = pd.DataFrame(all_results)
results_df.to_csv('../results/clustering/evaluation_summary.csv', index=False)
print(f"\n✓ Saved evaluation summary to ../results/clustering/evaluation_summary.csv")

# ===== FIND BEST CONFIGURATIONS =====
print("\n" + "=" * 70)
print("BEST CONFIGURATIONS")
print("=" * 70)

for method in ['Doc2Vec', 'Word2Vec_bins']:
    method_df = results_df[results_df['method'] == method]
    best_idx = method_df['silhouette'].idxmax()
    best = method_df.loc[best_idx]
    print(f"\n{method}:")
    print(f"  Best: dim={int(best['dim'])}, k={int(best['n_clusters'])}")
    print(f"  Silhouette={best['silhouette']:.4f}, DB={best['davies_bouldin']:.4f}, CH={best['calinski_harabasz']:.1f}")

print("\n" + "=" * 70)
