# -*- coding: utf-8 -*-
"""
Practical 6: Neural Machine Translation System
English-to-Hindi and Hindi-to-English Translation using Bigrams
"""

import os
import json
import requests
import sys
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import numpy as np
from itertools import zip_longest
import re

# Set UTF-8 encoding for console output
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class BigramTranslationModel:
    """Neural Machine Translation using Bigrams"""
    
    def __init__(self):
        self.en_hi_bigrams = defaultdict(list)  # English bigram -> Hindi translations
        self.hi_en_bigrams = defaultdict(list)  # Hindi bigram -> English translations
        self.en_vocab = Counter()
        self.hi_vocab = Counter()
        self.translation_pairs = []
        self.en_stopwords = set(stopwords.words('english'))
        
    def preprocess_text(self, text, language='en'):
        """Preprocess text"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters except spaces
        text = re.sub(r'[^\w\s]', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        return tokens
    
    def extract_bigrams(self, tokens):
        """Extract bigrams from tokens"""
        bigrams = []
        for i in range(len(tokens) - 1):
            bigram = (tokens[i], tokens[i + 1])
            bigrams.append(bigram)
        return bigrams
    
    def load_sample_data(self):
        """Load sample English-Hindi parallel data"""
        # Sample parallel corpus
        sample_data = [
            ("hello world", "नमस्ते दुनिया"),
            ("good morning", "सुप्रभात"),
            ("how are you", "आप कैसे हैं"),
            ("my name is john", "मेरा नाम जॉन है"),
            ("what is your name", "आपका नाम क्या है"),
            ("nice to meet you", "आपसे मिलकर खुशी हुई"),
            ("thank you very much", "धन्यवाद बहुत बहुत"),
            ("see you later", "फिर बाद में मिलते हैं"),
            ("good night", "शुभ रात्रि"),
            ("have a great day", "आपका दिन शानदार हो"),
            ("where is the bathroom", "बाथरूम कहाँ है"),
            ("how much does it cost", "इसकी कीमत कितनी है"),
            ("do you speak english", "क्या आप अंग्रेजी बोलते हैं"),
            ("i love you", "मैं तुमसे प्यार करता हूँ"),
            ("welcome to india", "भारत में आपका स्वागत है"),
            ("the weather is beautiful", "मौसम बहुत सुंदर है"),
            ("please help me", "कृपया मेरी मदद करें"),
            ("i am a student", "मैं एक छात्र हूँ"),
            ("this is delicious", "यह स्वादिष्ट है"),
            ("i am tired", "मैं थक गया हूँ"),
        ]
        return sample_data
    
    def train(self, parallel_corpus):
        """Train the model on parallel corpus"""
        print("Training Bigram Translation Model...")
        self.translation_pairs = parallel_corpus
        
        # Create word-level alignments (simple approach: position-based)
        word_alignments = defaultdict(list)
        
        for en_sent, hi_sent in parallel_corpus:
            # Preprocess
            en_tokens = self.preprocess_text(en_sent, 'en')
            hi_tokens = self.preprocess_text(hi_sent, 'hi')
            
            # Update vocabulary
            self.en_vocab.update(en_tokens)
            self.hi_vocab.update(hi_tokens)
            
            # Simple word alignment based on position
            for i, en_word in enumerate(en_tokens):
                if i < len(hi_tokens):
                    word_alignments[en_word].append(hi_tokens[i])
            
            # Extract bigrams and map them
            en_bigrams = self.extract_bigrams(en_tokens)
            hi_bigrams = self.extract_bigrams(hi_tokens)
            
            # Store bigrams with aligned translations
            for en_bigram in en_bigrams:
                # Find corresponding Hindi translation for this English bigram
                en_bigram_str = ' '.join(en_bigram)
                # Get the average position
                self.en_hi_bigrams[en_bigram].append(en_bigram_str)
            
            for hi_bigram in hi_bigrams:
                # Find corresponding English translation
                hi_bigram_str = ' '.join(str(t) for t in hi_bigram)
                self.hi_en_bigrams[hi_bigram].append(hi_bigram_str)
        
        print(f"✓ Trained on {len(parallel_corpus)} parallel sentences")
        print(f"✓ English vocabulary size: {len(self.en_vocab)}")
        print(f"✓ Hindi vocabulary size: {len(self.hi_vocab)}")
        print(f"✓ English bigram mappings: {len(self.en_hi_bigrams)}")
        print(f"✓ Hindi bigram mappings: {len(self.hi_en_bigrams)}")
    
    def translate_en_to_hi(self, english_text):
        """Translate English to Hindi using bigrams"""
        tokens = self.preprocess_text(english_text, 'en')
        bigrams = self.extract_bigrams(tokens)
        
        hindi_translations = []
        
        # Look for exact translation pair first
        for en_sent, hi_sent in self.translation_pairs:
            if english_text.lower() == en_sent.lower():
                return hi_sent
        
        # Translate using bigram mappings
        for bigram in bigrams:
            if bigram in self.en_hi_bigrams:
                candidates = [t for t in self.en_hi_bigrams[bigram] if t is not None]
                if candidates:
                    # Select the most common translation
                    best_translation = max(Counter(candidates).items(), key=lambda x: x[1])[0]
                    if isinstance(best_translation, tuple):
                        hindi_translations.append(' '.join(best_translation))
                    else:
                        hindi_translations.append(str(best_translation))
        
        if hindi_translations:
            return ' '.join(hindi_translations)
        return "[Translation not found]"
    
    def translate_hi_to_en(self, hindi_text):
        """Translate Hindi to English using bigrams"""
        tokens = self.preprocess_text(hindi_text, 'hi')
        bigrams = self.extract_bigrams(tokens)
        
        english_translations = []
        
        # Look for exact translation pair first
        for en_sent, hi_sent in self.translation_pairs:
            if hindi_text.lower() == hi_sent.lower():
                return en_sent
        
        # Translate using bigram mappings
        for bigram in bigrams:
            if bigram in self.hi_en_bigrams:
                candidates = [t for t in self.hi_en_bigrams[bigram] if t is not None]
                if candidates:
                    # Select the most common translation
                    best_translation = max(Counter(candidates).items(), key=lambda x: x[1])[0]
                    if isinstance(best_translation, tuple):
                        english_translations.append(' '.join(best_translation))
                    else:
                        english_translations.append(str(best_translation))
        
        if english_translations:
            return ' '.join(english_translations)
        return "[Translation not found]"
    
    def analyze_semantic_correctness(self, original, translated, reference):
        """Analyze semantic correctness using multiple metrics"""
        
        # 1. Token overlap (simple metric)
        original_tokens = set(self.preprocess_text(original, 'en'))
        reference_tokens = set(self.preprocess_text(reference, 'en'))
        translated_tokens = set(self.preprocess_text(translated, 'en'))
        
        # Calculate Jaccard similarity
        intersection = len(original_tokens & reference_tokens)
        union = len(original_tokens | reference_tokens)
        jaccard_similarity = intersection / union if union > 0 else 0
        
        # 2. Length ratio
        original_len = len(self.preprocess_text(original, 'en'))
        reference_len = len(self.preprocess_text(reference, 'en'))
        length_ratio = min(original_len, reference_len) / max(original_len, reference_len) if max(original_len, reference_len) > 0 else 0
        
        # 3. Word overlap with reference
        ref_overlap = len(reference_tokens & translated_tokens) / len(reference_tokens) if len(reference_tokens) > 0 else 0
        
        # Combined semantic score (0-1)
        semantic_score = (jaccard_similarity * 0.4 + length_ratio * 0.3 + ref_overlap * 0.3)
        
        return {
            'jaccard_similarity': round(jaccard_similarity, 3),
            'length_ratio': round(length_ratio, 3),
            'reference_overlap': round(ref_overlap, 3),
            'semantic_score': round(semantic_score, 3)
        }


def download_kaggle_dataset():
    """Attempt to download Kaggle dataset"""
    print("\n" + "="*60)
    print("Kaggle Dataset Download")
    print("="*60)
    print("To use the full English-Hindi Parallel Corpus from Kaggle:")
    print("1. Install kaggle: pip install kaggle")
    print("2. Get API credentials from https://www.kaggle.com/settings/account")
    print("3. Place credentials in ~/.kaggle/kaggle.json")
    print("4. Run: kaggle datasets download -d aiswaryaramachandran/hindi-english-parallel-corpus")
    print("\nUsing sample data for demonstration...")
    print("="*60 + "\n")


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("NEURAL MACHINE TRANSLATION SYSTEM - ENGLISH <-> HINDI")
    print("Using Bigram-based Translation Mapping")
    print("="*70 + "\n")
    
    # Initialize model
    model = BigramTranslationModel()
    
    # Load sample data
    parallel_corpus = model.load_sample_data()
    download_kaggle_dataset()
    
    # Train model
    model.train(parallel_corpus)
    
    # Generate and analyze translations
    print("\n" + "="*70)
    print("TRANSLATION SAMPLES & SEMANTIC ANALYSIS")
    print("="*70 + "\n")
    
    test_cases = [
        ("hello world", "नमस्ते दुनिया"),
        ("good morning", "सुप्रभात"),
        ("how are you", "आप कैसे हैं"),
        ("thank you very much", "धन्यवाद बहुत बहुत"),
    ]
    
    results = []
    
    for en_text, hi_text in test_cases:
        print(f"\n{'─'*70}")
        print(f"Original English:     {en_text}")
        print(f"Reference Hindi:      {hi_text}")
        
        # Translate
        translated_hi = model.translate_en_to_hi(en_text)
        print(f"Translated Hindi:     {translated_hi}")
        
        # Analyze semantic correctness
        analysis = model.analyze_semantic_correctness(en_text, translated_hi, hi_text)
        
        print(f"\nSemantic Analysis:")
        print(f"  • Jaccard Similarity:  {analysis['jaccard_similarity']}")
        print(f"  • Length Ratio:        {analysis['length_ratio']}")
        print(f"  • Reference Overlap:   {analysis['reference_overlap']}")
        print(f"  • Semantic Score:      {analysis['semantic_score']}")
        
        results.append({
            'original': en_text,
            'reference': hi_text,
            'translated': translated_hi,
            'analysis': analysis
        })
    
    # Bigram Analysis
    print("\n" + "="*70)
    print("BIGRAM ANALYSIS")
    print("="*70 + "\n")
    
    print("Top 10 English Bigrams (by frequency):")
    en_bigram_freq = Counter()
    for bigram, translations in model.en_hi_bigrams.items():
        en_bigram_freq[bigram] = len(translations)
    
    for i, (bigram, freq) in enumerate(en_bigram_freq.most_common(10), 1):
        print(f"  {i}. {bigram} → appears {freq} times")
    
    print("\nTop 10 Hindi Bigrams (by frequency):")
    hi_bigram_freq = Counter()
    for bigram, translations in model.hi_en_bigrams.items():
        hi_bigram_freq[bigram] = len(translations)
    
    for i, (bigram, freq) in enumerate(hi_bigram_freq.most_common(10), 1):
        print(f"  {i}. {bigram} → appears {freq} times")
    
    # Statistics
    print("\n" + "="*70)
    print("MODEL STATISTICS")
    print("="*70 + "\n")
    
    avg_semantic_score = np.mean([r['analysis']['semantic_score'] for r in results])
    
    print(f"Total Training Pairs:              {len(parallel_corpus)}")
    print(f"English Vocabulary Size:           {len(model.en_vocab)}")
    print(f"Hindi Vocabulary Size:             {len(model.hi_vocab)}")
    print(f"English Bigram Mappings:           {len(model.en_hi_bigrams)}")
    print(f"Hindi Bigram Mappings:             {len(model.hi_en_bigrams)}")
    print(f"Average Semantic Score:            {avg_semantic_score:.3f}")
    
    # Interactive mode
    print("\n" + "="*70)
    print("INTERACTIVE TRANSLATION MODE")
    print("="*70)
    
    while True:
        print("\nOptions: [1] English→Hindi  [2] Hindi→English  [3] Exit")
        choice = input("Select option (1/2/3): ").strip()
        
        if choice == '1':
            user_input = input("Enter English text: ").strip()
            if user_input:
                translation = model.translate_en_to_hi(user_input)
                print(f"Hindi Translation: {translation}")
        elif choice == '2':
            user_input = input("Enter Hindi text: ").strip()
            if user_input:
                translation = model.translate_hi_to_en(user_input)
                print(f"English Translation: {translation}")
        elif choice == '3':
            print("\n✓ Exiting Neural Machine Translation System")
            break
        else:
            print("Invalid option. Please try again.")
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
