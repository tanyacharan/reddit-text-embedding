# 1_prepare_fixed.py - Properly handle missing/empty data
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import numpy as np

print("Running FIXED 1_prepare.py")

# Download stopwords
nltk.download('stopwords', quiet=True)
stop = set(stopwords.words('english'))

# Read CSV
df = pd.read_csv(
    '../data/reddit_posts.csv',
    sep=None,
    engine='python',
    on_bad_lines='skip',
    names=['text'],
    header=None,
    encoding='utf-8'
)
print(f"Loaded file successfully. Initial rows: {len(df)}")

# Check for missing data
print(f"Missing values: {df['text'].isna().sum()}")
print(f"Empty strings: {(df['text'] == '').sum()}")

# Drop rows with missing or empty text BEFORE processing
df = df.dropna(subset=['text'])
df = df[df['text'].str.strip() != '']
print(f"After removing empty rows: {len(df)}")

def clean_text(text):
    """Clean and tokenize text, handling edge cases."""
    # Don't convert None/NaN - we already filtered them out
    if not isinstance(text, str):
        return []
    
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove special characters, keep only alphanumeric and spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Tokenize
    tokens = text.split()
    
    # Filter: remove stopwords and short tokens
    tokens = [t for t in tokens if t not in stop and len(t) > 1]
    
    return tokens

# Apply cleaning
df['tokens'] = df['text'].apply(clean_text)

# Filter out documents that ended up with no tokens after cleaning
initial_count = len(df)
df = df[df['tokens'].apply(lambda x: len(x) > 0)]
print(f"After removing empty token lists: {len(df)} (removed {initial_count - len(df)})")

# Final check for garbage
def has_only_garbage(tokens):
    """Check if tokens contain only garbage values."""
    if len(tokens) == 0:
        return True
    garbage = {'none', 'nan', 'null'}
    return all(t in garbage for t in tokens)

garbage_count = df['tokens'].apply(has_only_garbage).sum()
if garbage_count > 0:
    print(f"⚠️  WARNING: Found {garbage_count} documents with garbage tokens")
    df = df[~df['tokens'].apply(has_only_garbage)]
    print(f"After removing garbage: {len(df)}")

# Save cleaned data
df.to_pickle('../data/reddit_posts_prepped.pkl')
print(f"\n✅ Saved cleaned data to ../data/reddit_posts_prepped.pkl")
print(f"Final document count: {len(df)}")

# Show sample
print("\n--- Sample cleaned documents ---")
for idx, row in df.head(3).iterrows():
    print(f"  {row['tokens'][:15]}...")
