from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
from tqdm import tqdm
import joblib
import os

df = pd.read_pickle('../data/reddit_posts_prepped.pkl')
documents = [TaggedDocument(words=tokens, tags=[str(i)]) for i,tokens in enumerate(df['tokens'])]

vector_sizes = [50, 100, 300]
for vs in vector_sizes:
    model_path = f'../results/doc2vec_embeddings/doc2vec_vs{vs}.model'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model = Doc2Vec(vector_size=vs, window=5, min_count=2, workers=4, epochs=20)
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(model_path)
    # generate doc vectors
    vectors = [model.infer_vector(doc.words, epochs=20) for doc in documents]
    joblib.dump(vectors, f'../results/doc2vec_embeddings/docvecs_vs{vs}.pkl')
    print("Saved vs", vs)
