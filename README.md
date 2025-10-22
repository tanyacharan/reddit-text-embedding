# Lab 8: Document Embeddings & Clustering Analysis

Comprehensive comparison of **Doc2Vec** vs **Word2Vec + Binning** for document representation and clustering on Reddit posts dataset.

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Directory Structure](#directory-structure)
- [Pipeline Scripts](#pipeline-scripts)
- [Results & Visualizations](#results--visualizations)

---

## Overview

This project implements and compares two document embedding approaches:

1. **Doc2Vec**: Direct document-to-vector learning using Gensim's Doc2Vec
2. **Word2Vec + Binning**: Words clustered into bins, documents represented as bin frequency vectors

**Evaluation Approach**:
- 3 embedding dimensions: 50, 100, 300
- Multiple cluster counts: k = 4, 5, 8, 10, 15
- 3 metrics: Silhouette, Davies-Bouldin, Calinski-Harabasz
- Both quantitative (metrics) and qualitative (manual inspection) analysis

---

## Requirements

### Python Version
- Python 3.11+

### Dependencies
```bash
pip install -r requirements.txt
```

### Data
- `reddit_posts.csv` in `../data/` directory
- CSV should contain Reddit post text (one post per row)

---

## Quick Start

```bash
# Navigate to src directory
cd src/

# Run complete pipeline
python 1_prepare_fixed.py           # Clean data
python 2_doc2vec.py                 # Generate Doc2Vec embeddings
python 3_word2vec_binning.py        # Generate Word2Vec binning embeddings
python 4_cluster_eval_fixed.py      # Cluster & evaluate both methods
python 5_inspect_clusters_robust.py # Qualitative analysis
python 6_visualize.py               # Create visualizations

# Results will be in ../comparison_results/plots/
```

---

## Directory Structure

```
project_root/
│
├── src/                             
│   ├── 1_prepare_fixed.py           
│   ├── 2_doc2vec.py                 
│   ├── 3_word2vec_binning.py        
│   ├── 4_cluster_eval_fixed.py      
│   ├── 5_inspect_clusters_robust.py 
│   ├── 6_visualize.py               
│   └── diagnostic_check_data.py     
│
├── data/
│   ├── reddit_posts.csv             # Raw input 
│   └── reddit_posts_prepped.pkl     # Cleaned data (generated in 1.py)
│
├── results/
│   ├── doc2vec_embeddings/          # Doc2Vec models & vectors
│   ├── word2vec/                    # Word2Vec model
│   ├── word2vec_bins/               # Binned embeddings (K50, K100, K300)
│   └── clustering/                  # Clustering results
│       └── evaluation_summary.csv   # Main Results
│
├── comparison_results/              # Final Outputs
│   ├── plots/                       # All visualizations (12+ images)
│   └── evaluation_summary_with_rankings.csv
│
├── requirements.txt
└── README.md                        
```

---

## Pipeline Scripts

### `1_prepare_fixed.py` - Data Preprocessing

**Input**: `../data/reddit_posts.csv`  
**Output**: `../data/reddit_posts_prepped.pkl`

- Cleans text, tokenizes, removes stopwords
- Properly handles `NaN`/`None` values instead of `str(text).lower()`

---

### `2_doc2vec.py` - Doc2Vec Embeddings

**Input**: `../data/reddit_posts_prepped.pkl`  
**Output**: `../results/doc2vec_embeddings/`

- Trains 3 Doc2Vec models (vector_size = 50, 100, 300)
- 20 epochs per model

---

### `3_word2vec_binning.py` - Word2Vec + Binning

**Input**: `../data/reddit_posts_prepped.pkl`  
**Output**: `../results/word2vec_bins/K{50,100,300}/`

- Trains Word2Vec on vocabulary
- Clusters words into K bins (K = 50, 100, 300)
- Creates document vectors as normalized bin frequencies

---

### `4_cluster_eval_fixed.py` - Clustering & Evaluation

**Input**: All embeddings from steps 2 & 3  
**Output**: `../results/clustering/evaluation_summary.csv`

- Clusters with KMeans (k = 4, 5, 8, 10, 15)
- Tests both methods (30 total configurations)
- Computes 3 metrics: Silhouette, Davies-Bouldin, Calinski-Harabasz

---

### `5_inspect_clusters_robust.py` - Qualitative Analysis

**Input**: Cluster labels + original data  
**Output**: Console output with sample documents

- Shows sample documents from each cluster
- Allows manual assessment of semantic coherence

---

### `6_visualize.py` - Comprehensive Visualizations
**Input**: `evaluation_summary.csv` + embeddings  
**Output**: `../comparison_results/plots/` (12+ images)

Creates 6 types of visualizations:
1. **PCA scatter plots** - Best configurations
2. **Metric comparison bars** - Side-by-side performance
3. **Heatmaps** - Full parameter space (6 heatmaps)
4. **Side-by-side comparison** - Direct visual showdown
5. **Performance summary** - Overall winner
6. **Dimensionality effect** - Impact of dimension

---

## Results & Visualizations

### Key Output Files

1. **`evaluation_summary.csv`** - All configurations with metrics
2. **`evaluation_summary_with_rankings.csv`** - Enhanced with rankings
3. **Plots** (in `comparison_results/plots/`):
   - `metric_comparison_bars.png`
   - `side_by_side_best.png`
   - `performance_summary.png`
   - `heatmap_*.png` (6 files)
   - Individual PCA plots

### Interpreting Metrics

**Silhouette Score** (0 to 1, higher = better)

- 0.8+ = Excellent
- 0.7-0.8 = Good
- 0.5-0.7 = Moderate
- <0.5 = Poor

**Davies-Bouldin** (0 to ∞, lower = better)

- <1.0 = Excellent
- 1.0-2.0 = Good
- \>2.0 = Poor

**Calinski-Harabasz** (0 to ∞, higher = better)

- Compare relative values between methods

### Empirical Best Results

| Method | Best Dim | Best k | Silhouette | DB | CH |
|--------|----------|--------|------------|----|----|
| Doc2Vec | 300 | 4 | 0.2075 | 2.3327 | 235.6 |
| Word2Vec_bins | 50 | 4 | 0.1334 | 3.2499 | 167.3 |

**Winner**: Doc2Vec (surprising given Word2Vec + Binning should normally perform?)

---

Disclaimer: The collected Reddit data is to be used for educational purposes only. Please do not sue us, thanks.