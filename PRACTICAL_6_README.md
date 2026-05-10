# Practical 6: Neural Machine Translation System (English ↔ Hindi)

## Overview
This practical implements a Neural Machine Translation (NMT) system that uses **bigram-based translation mappings** to perform English-to-Hindi and Hindi-to-English translation. The system learns translation patterns from parallel corpora and generates translations while analyzing semantic correctness.

## Project Objectives
1. Build a bigram-based translation model
2. Map bigrams to translations in both directions
3. Generate translation samples
4. Analyze semantic correctness of translations
5. Support interactive translation mode

## Architecture & Components

### 1. **BigramTranslationModel** Class
Core class implementing the NMT system with the following components:

#### a. **Text Preprocessing** (`preprocess_text`)
- Converts text to lowercase
- Removes special characters (keeps only alphanumeric and spaces)
- Tokenizes using NLTK `word_tokenize`
- Handles both English and Hindi text

#### b. **Bigram Extraction** (`extract_bigrams`)
- Creates consecutive word pairs from token sequences
- For a sequence [w1, w2, w3, w4], generates [(w1,w2), (w2,w3), (w3,w4)]
- Essential for capturing word-order patterns in translation

#### c. **Training** (`train`)
**Process:**
1. Load parallel corpus (English-Hindi sentence pairs)
2. Preprocess both source and target sentences
3. Update vocabulary counters for both languages
4. Extract bigrams from each language
5. Create position-based word alignments
6. Store bigram-to-translation mappings in both directions

**Output Metrics:**
- Training pairs count
- Vocabulary sizes (English & Hindi)
- Bigram mapping counts

#### d. **Translation Generation**
**English → Hindi** (`translate_en_to_hi`):
1. First checks for exact sentence matches in training data
2. If found, returns exact translation
3. Otherwise, preprocesses input and extracts bigrams
4. For each bigram, retrieves candidate translations
5. Selects most common translation for each bigram
6. Concatenates translations into output

**Hindi → English** (`translate_hi_to_en`):
- Same process but reversed (Hindi input → English output)
- Looks up Hindi bigrams in bidirectional mappings

#### e. **Semantic Correctness Analysis** (`analyze_semantic_correctness`)
Evaluates translation quality using multiple metrics:

**Metric 1: Jaccard Similarity**
- Formula: |A ∩ B| / |A ∪ B|
- Compares token sets between original and reference
- Range: 0.0 to 1.0

**Metric 2: Length Ratio**
- Compares token counts between original and reference
- Formula: min(len1, len2) / max(len1, len2)
- Penalizes significantly different sentence lengths
- Range: 0.0 to 1.0

**Metric 3: Reference Overlap**
- Measures token overlap between translation and reference
- Formula: |tokens_translated ∩ tokens_reference| / |tokens_reference|
- Indicates how many reference tokens appear in translation
- Range: 0.0 to 1.0

**Composite Semantic Score**
- Weighted combination of three metrics:
  - Jaccard Similarity: 40% weight
  - Length Ratio: 30% weight
  - Reference Overlap: 30% weight
- Formula: 0.4×J + 0.3×L + 0.3×O
- Range: 0.0 to 1.0

### 2. **Sample Parallel Corpus**
20 English-Hindi sentence pairs covering common phrases:
- Greetings: "hello world" ↔ "नमस्ते दुनिया"
- Polite expressions: "nice to meet you" ↔ "आपसे मिलकर खुशी हुई"
- Questions: "how are you" ↔ "आप कैसे हैं"
- Status expressions: "I am tired" ↔ "मैं थक गया हूँ"

### 3. **Bigram Analysis**
The system generates frequency analysis of:
- Top 10 English bigrams and their occurrence counts
- Top 10 Hindi bigrams and their occurrence counts
- Shows which word pairs are most commonly aligned

### 4. **Interactive Translation Mode**
Menu-driven interface supporting:
- Option 1: English → Hindi translation
- Option 2: Hindi → English translation
- Option 3: Exit program
- Real-time user input and translation output

## Key Features

### Multi-Directional Translation
- Bidirectional support (English ↔ Hindi)
- Uses aligned bigram mappings for both directions
- Fallback to exact sentence matches for training data

### Semantic Analysis
- Automatically scores translation quality
- Provides detailed breakdown of similarity metrics
- Computes aggregate semantic correctness score
- Useful for evaluating translation system performance

### Statistical Reporting
- Training statistics (corpus size, vocabulary sizes)
- Bigram frequency analysis
- Average semantic score across test set
- Vocabulary coverage metrics

### Extensibility
- Easy to add new parallel corpus data
- Modular design allows upgrading translation algorithms
- Can integrate actual Kaggle dataset with minimal changes

## Usage

### Basic Execution
```bash
python practical_6.py
```

### Running Translation Tasks
1. Program starts with model training on sample data
2. Displays translation samples with semantic analysis
3. Shows bigram statistics and frequency analysis
4. Enters interactive mode for user translations

### Sample Output
```
Original English: hello world
Reference Hindi: नमस्ते दुनिया
Translated Hindi: नमस्ते दुनिया

Semantic Analysis:
  • Jaccard Similarity: 0.0
  • Length Ratio: 1.0
  • Reference Overlap: 1.0
  • Semantic Score: 0.6
```

## Performance Metrics

### Model Statistics
- **Total Training Pairs**: 20
- **English Vocabulary Size**: 49
- **Hindi Vocabulary Size**: 48
- **English Bigram Mappings**: 47
- **Hindi Bigram Mappings**: 53
- **Average Semantic Score**: ~0.54

### Semantic Score Interpretation
- **0.9-1.0**: Excellent translation (nearly perfect semantic preservation)
- **0.7-0.9**: Good translation (most meaning preserved)
- **0.5-0.7**: Fair translation (reasonable semantic overlap)
- **0.3-0.5**: Poor translation (limited semantic correctness)
- **0.0-0.3**: Very poor translation (minimal semantic overlap)

## Kaggle Dataset Integration

### To Use Full English-Hindi Parallel Corpus

1. **Install Kaggle API**:
   ```bash
   pip install kaggle
   ```

2. **Get API Credentials**:
   - Visit https://www.kaggle.com/settings/account
   - Scroll to "API" section and click "Create New API Token"
   - Place the downloaded `kaggle.json` in `~/.kaggle/` directory

3. **Download Dataset**:
   ```bash
   kaggle datasets download -d aiswaryaramachandran/hindi-english-parallel-corpus
   unzip hindi-english-parallel-corpus.zip
   ```

4. **Update Code**:
   Replace `load_sample_data()` with actual dataset loading logic

## Limitations & Future Improvements

### Current Limitations
1. **Simple Alignment**: Uses basic position-based word alignment
2. **Exact Match Fallback**: Requires exact sentence matches for accuracy
3. **Limited Corpus**: Sample data has only 20 sentences
4. **No Morphological Analysis**: Doesn't handle inflections or word forms
5. **No Context**: Each bigram is translated independently

### Potential Improvements
1. **Better Alignment Algorithm**:
   - Implement GIZA++ or Bayesian alignment
   - Use statistical machine translation techniques

2. **Advanced NMT Architecture**:
   - Sequence-to-sequence models with attention
   - Transformer-based models (BERT, mT5)
   - Byte-pair encoding (BPE) for subword units

3. **Larger Training Corpus**:
   - Integrate full Kaggle dataset (100K+ pairs)
   - Use pre-trained embeddings (fastText, mBERT)

4. **Context Awareness**:
   - Implement n-grams (trigrams, 4-grams)
   - Add language models for better context

5. **Evaluation Metrics**:
   - Implement BLEU score
   - Add manual evaluation framework
   - Compute METEOR or CIDEr scores

## Dependencies
- **nltk**: Tokenization and stopword filtering
- **numpy**: Numerical computations
- **python 3.7+**: Core language

### Installation
```bash
pip install nltk numpy
```

## Project Structure
```
practical_6.py              # Main NMT system
PRACTICAL_6_README.md      # This documentation
requirements.txt           # Project dependencies
```

## References
1. Och, F. J., & Ney, H. (2003). "A Systematic Comparison of Various Statistical Alignment Models"
2. Koehn, P. (2009). "Statistical Machine Translation"
3. Vaswani, A., et al. (2017). "Attention is All You Need" (Transformer paper)

## Notes
- This implementation uses bigrams for demonstration purposes
- Production systems typically use deep neural networks
- Consider using pre-trained multilingual models (mT5, mBART) for better results
- Larger datasets (100K+) typically show much better semantic correctness
