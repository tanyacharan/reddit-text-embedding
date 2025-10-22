# src/3_word2vec_binning.py
from gensim.models import Word2Vec
import joblib
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import os

df = pd.read_pickle('../data/reddit_posts_prepped.pkl')
sentences = df['tokens'].tolist()
import os
os.makedirs('../results/word2vec', exist_ok=True)
# Train Word2Vec
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4, epochs=10)
w2v_model.save('../results/word2vec/w2v.model')

# Get word vectors and vocabulary
words = list(w2v_model.wv.index_to_key)
word_vecs = np.array([w2v_model.wv[word] for word in words])

# We'll create binning for each target dimensionality
target_dims = [50,100,300]   # use these as K for word bins
for K in target_dims:
    km = KMeans(n_clusters=K, random_state=42, n_init=10)
    km.fit(word_vecs)
    word_to_bin = {word:int(bin_) for word, bin_ in zip(words, km.labels_)}
    os.makedirs(f'../results/word2vec_bins/K{K}', exist_ok=True)
    joblib.dump(word_to_bin, f'../results/word2vec_bins/K{K}/word2bin.pkl')
    joblib.dump(km.cluster_centers_, f'../results/word2vec_bins/K{K}/centers.pkl')
    print("Saved bins for K=", K)

    # create document vectors: normalized frequency per bin
    doc_vectors = []
    for tokens in sentences:
        counts = np.zeros(K, dtype=float)
        for t in tokens:
            if t in word_to_bin:
                counts[word_to_bin[t]] += 1
        if counts.sum() == 0:
            doc_vectors.append(counts)  # zero vector
        else:
            doc_vectors.append(counts / counts.sum())
    joblib.dump(doc_vectors, f'../results/word2vec_bins/K{K}/docvecs.pkl')
    print("Saved doc vectors for K", K)
