import pandas as pd
import time
import re
import nltk
from nltk.tokenize import WhitespaceTokenizer, WordPunctTokenizer, TreebankWordTokenizer, TweetTokenizer
from collections import Counter
import matplotlib.pyplot as plt

# Ensure NLTK resources are downloaded (if any needed for future expansions)
nltk.download('punkt', quiet=True)

def load_data(filepath='IMDB Dataset.csv', sample_size=1000):
    """Loads a sample of the IMDb dataset."""
    df = pd.read_csv(filepath)
    print(f"Loaded {filepath} successfully.")
    # Take a sample to reduce processing time for demonstration
    return df['review'].head(sample_size).tolist()

def analyze_tokenization(texts, tokenizer_name, tokenize_func):
    """Applies tokenization and calculates statistics."""
    start_time = time.time()
    
    all_tokens = []
    for text in texts:
        tokens = tokenize_func(text)
        all_tokens.extend(tokens)
        
    processing_time = time.time() - start_time
    vocab = set(all_tokens)
    vocab_size = len(vocab)
    freq_dist = Counter(all_tokens)
    
    return {
        'name': tokenizer_name,
        'time': processing_time,
        'vocab_size': vocab_size,
        'total_tokens': len(all_tokens),
        'top_10_words': freq_dist.most_common(10)
    }

def main():
    print("--- Practical 1: Tokenization Comparative Analysis ---\n")
    texts = load_data(sample_size=5000)
    
    # Initialize tokenizers
    whitespace_tokenizer = WhitespaceTokenizer()
    punct_tokenizer = WordPunctTokenizer()
    treebank_tokenizer = TreebankWordTokenizer()
    tweet_tokenizer = TweetTokenizer()
    
    # Define tokenization functions
    tokenizers = [
        ("Whitespace", whitespace_tokenizer.tokenize),
        ("Punctuation-based", punct_tokenizer.tokenize),
        ("Treebank", treebank_tokenizer.tokenize),
        ("Tweet", tweet_tokenizer.tokenize)
    ]
    
    results = []
    for name, func in tokenizers:
        print(f"Running {name} Tokenizer...")
        stats = analyze_tokenization(texts, name, func)
        results.append(stats)
        
    print("\n--- Comparative Analysis Report ---")
    print(f"{'Tokenizer':<20} | {'Vocab Size':<12} | {'Processing Time (s)':<20} | {'Total Tokens':<15}")
    print("-" * 75)
    for res in results:
        print(f"{res['name']:<20} | {res['vocab_size']:<12} | {res['time']:<20.4f} | {res['total_tokens']:<15}")
        
    # Plotting processing time
    names = [r['name'] for r in results]
    times = [r['time'] for r in results]
    vocab_sizes = [r['vocab_size'] for r in results]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(names, times, color=['blue', 'orange', 'green', 'red'])
    plt.title('Processing Time Comparison')
    plt.ylabel('Time (seconds)')
    
    plt.subplot(1, 2, 2)
    plt.bar(names, vocab_sizes, color=['blue', 'orange', 'green', 'red'])
    plt.title('Vocabulary Size (Reduction) Comparison')
    plt.ylabel('Unique Tokens')
    
    plt.tight_layout()
    plt.savefig('practical_1_comparison.png')
    print("\nSaved comparison charts to 'practical_1_comparison.png'")
    
    print("\nTop 10 Word Frequencies:")
    for res in results:
        print(f"\n{res['name']}:")
        for word, count in res['top_10_words']:
            print(f"  {word}: {count}")

if __name__ == "__main__":
    main()
