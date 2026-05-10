# Practical 5: Question Answering Chatbot via Similarity Search

## Overview
This practical implements a Question Answering chatbot system that uses TF-IDF vectorization and cosine similarity search to find the most relevant answers to user queries from a Q&A dataset.

## Features Implemented

### 1. **Dataset Management** (`QADatasetLoader`)
- Load Q&A pairs from SQuAD dataset or create sample dataset
- Sample dataset includes 15 Q&A pairs covering ML/NLP concepts
- Support for custom dataset loading from JSON format

### 2. **Similarity Search Engine** (`SimilaritySearchEngine`)
- Uses **TF-IDF vectorization** with:
  - Lowercase normalization
  - English stopword removal
  - Up to 5000 features
  - N-gram support (unigrams + bigrams)
- Computes **cosine similarity** between query and all questions
- Returns top-k most similar Q&A pairs with confidence scores

### 3. **QA Chatbot** (`QAChatbot`)
- Answers queries using similarity search
- Maintains conversation history
- Interactive chat mode
- Batch evaluation capabilities
- Confidence threshold support

### 4. **Evaluation Metrics** (`QAEvaluator`)

#### a. **Exact Match (EM) Score**
- Binary metric: 1 if prediction exactly matches reference, 0 otherwise
- After normalization (lowercase, whitespace handling)
- Use case: Measuring perfect answer matches

#### b. **F1 Score**
- Token-based overlap metric
- Combines precision and recall
- Formula: F1 = 2 × (precision × recall) / (precision + recall)
- Uses normalized tokens (removes articles, punctuation)
- Range: 0.0 to 1.0

#### c. **BLEU Score**
- Machine translation metric adapted for QA
- Measures n-gram overlap with reference
- Uses 1-gram and 2-gram weights (0.5 each)
- Includes smoothing function to avoid zero scores
- Range: 0.0 to 1.0

### 5. **Text Normalization**
- Removes articles (a, an, the)
- Removes punctuation
- Lowercase conversion
- Whitespace normalization

## Usage Examples

### Basic Query
```python
chatbot = QAChatbot(qa_pairs)
response, score, matched_question = chatbot.answer("What is machine learning?")
```

### Batch Evaluation
```python
evaluation_summary = chatbot.evaluate_on_test_set(test_questions, test_answers)
print(f"Exact Match: {evaluation_summary['exact_match_avg']:.4f}")
print(f"F1 Score: {evaluation_summary['f1_score_avg']:.4f}")
print(f"BLEU Score: {evaluation_summary['bleu_score_avg']:.4f}")
```

### Interactive Chat
```python
chatbot.interactive_chat()
# User types queries, gets responses with confidence scores
```

## Test Results

When run on the sample dataset:
- **Exact Match Score**: 1.0000 (perfect for identical questions)
- **F1 Score**: 1.0000 (perfect token overlap)
- **BLEU Score**: 1.0000 (perfect n-gram overlap)
- **Query Matching Speed**: Near-instantaneous
- **Similarity Detection**: Successfully identifies related questions with varying confidence scores

## Example Metrics Demonstration

For prediction "Machine learning is a subset of artificial intelligence" vs 
reference "Machine learning is a subset of AI that enables computers to learn from data":

- **Exact Match**: 0.0000 (strings not identical)
- **F1 Score**: 0.5000 (50% token overlap after normalization)
- **BLEU Score**: 0.2928 (partial n-gram match)

## Dependencies

```
pandas
numpy
nltk
scikit-learn
```

## How to Run

```bash
python practical_5.py
```

The script will:
1. Load/create Q&A dataset
2. Initialize similarity search engine
3. Test with sample queries
4. Demonstrate evaluation metrics
5. Run batch evaluation
6. Optionally start interactive chat

## Architecture Insights

**Similarity Search Process:**
1. Vectorize all questions using TF-IDF
2. Vectorize incoming query
3. Compute cosine similarity with all question vectors
4. Return answer from question with highest similarity

**Evaluation Strategy:**
- Use multiple metrics for comprehensive evaluation
- Exact Match: Strict matching
- F1 Score: Flexible token-based matching
- BLEU Score: N-gram based similarity

## Advantages

✓ Fast similarity search using TF-IDF
✓ No external API dependencies
✓ Multiple evaluation metrics
✓ Conversational history tracking
✓ Confidence scores for transparency
✓ Handles typos and variations via token matching

## Limitations & Future Enhancements

- Limited to Q&A pairs in dataset (no open-ended generation)
- TF-IDF doesn't capture semantic meaning well (could use embeddings)
- No context awareness between turns
- Could implement:
  - BERT embeddings for semantic similarity
  - Multi-turn conversation context
  - Intent classification
  - Named entity recognition
  - Question paraphrasing

## Files

- `practical_5.py`: Main implementation
- This summary document

---
Created as part of NLP Practical Assignments
Dataset: SQuAD (Stanford Question Answering Dataset)
