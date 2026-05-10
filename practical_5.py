"""
Practical 5: Question Answering Chatbot via Similarity Search and Rules
Tasks:
1. Create a sub dataset for Q&A pairs
2. Run similarity search for queries across questions
3. Respond using the answer of the most similar Q&A pair
4. Evaluate using: Exact Match Score, F1 Score, BLEU Score
Dataset: SQuAD (Stanford Question Answering Dataset)
"""

import json
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# ============================================================================
# 1. DATASET LOADER - Create Q&A pairs dataset
# ============================================================================

class QADatasetLoader:
    """Load and prepare Q&A dataset from SQuAD or create sample dataset"""
    
    @staticmethod
    def create_sample_dataset(size=100):
        """Create a sample Q&A dataset for demonstration"""
        sample_qa = [
            {
                "question": "What is machine learning?",
                "answer": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed."
            },
            {
                "question": "What is deep learning?",
                "answer": "Deep learning is a subset of machine learning based on artificial neural networks with multiple layers."
            },
            {
                "question": "How does neural networks work?",
                "answer": "Neural networks consist of interconnected nodes (neurons) organized in layers that process information through weighted connections."
            },
            {
                "question": "What is natural language processing?",
                "answer": "Natural language processing is a branch of artificial intelligence that focuses on enabling computers to understand and process human language."
            },
            {
                "question": "What are transformers in NLP?",
                "answer": "Transformers are deep learning models that use attention mechanisms to process sequences of data efficiently."
            },
            {
                "question": "What is BERT?",
                "answer": "BERT is a transformer-based model pre-trained using bidirectional training of transformers for representation learning."
            },
            {
                "question": "What is fine-tuning?",
                "answer": "Fine-tuning is the process of adapting a pre-trained model to a specific task by training it with task-specific data."
            },
            {
                "question": "What is transfer learning?",
                "answer": "Transfer learning is a machine learning technique where knowledge gained from one task is applied to another related task."
            },
            {
                "question": "What is semantic similarity?",
                "answer": "Semantic similarity measures how alike two pieces of text are in meaning, often using embeddings or distance metrics."
            },
            {
                "question": "What is cosine similarity?",
                "answer": "Cosine similarity is a metric that measures the cosine of the angle between two vectors, ranging from -1 to 1."
            },
            {
                "question": "What is TF-IDF?",
                "answer": "TF-IDF is a numerical statistic that reflects how important a word is to a document in a collection of documents."
            },
            {
                "question": "What is tokenization?",
                "answer": "Tokenization is the process of breaking down text into smaller units called tokens, usually words or subwords."
            },
            {
                "question": "What is stemming?",
                "answer": "Stemming is a process of reducing words to their root form by removing prefixes or suffixes."
            },
            {
                "question": "What is lemmatization?",
                "answer": "Lemmatization is the process of reducing words to their base or dictionary form considering the morphological analysis."
            },
            {
                "question": "What is stop words?",
                "answer": "Stop words are common words that usually carry less meaningful information and are often removed during text processing."
            },
        ]
        return sample_qa
    
    @staticmethod
    def load_squad_dataset(filepath=None, limit=None):
        """Load SQuAD dataset from JSON file"""
        try:
            if filepath is None:
                print("SQuAD dataset not provided. Using sample dataset instead.")
                return QADatasetLoader.create_sample_dataset(limit or 100)
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            qa_pairs = []
            for article in data['data']:
                for paragraph in article['paragraphs']:
                    for qa in paragraph['qas']:
                        if qa['answers']:
                            qa_pairs.append({
                                'question': qa['question'],
                                'answer': qa['answers'][0]['text']
                            })
            
            if limit:
                qa_pairs = qa_pairs[:limit]
            
            return qa_pairs
        except Exception as e:
            print(f"Error loading SQuAD dataset: {e}")
            print("Using sample dataset instead.")
            return QADatasetLoader.create_sample_dataset(limit or 100)


# ============================================================================
# 2. SIMILARITY SEARCH ENGINE
# ============================================================================

class SimilaritySearchEngine:
    """Search for most similar Q&A pairs using TF-IDF and cosine similarity"""
    
    def __init__(self, qa_pairs):
        """Initialize with Q&A pairs"""
        self.qa_pairs = qa_pairs
        self.questions = [pair['question'] for pair in qa_pairs]
        self.answers = [pair['answer'] for pair in qa_pairs]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )
        
        # Fit vectorizer on all questions
        self.question_vectors = self.vectorizer.fit_transform(self.questions)
    
    def search(self, query, top_k=1):
        """Search for most similar question to the query"""
        # Vectorize the query
        query_vector = self.vectorizer.transform([query])
        
        # Compute cosine similarity with all questions
        similarities = cosine_similarity(query_vector, self.question_vectors)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return results with scores
        results = []
        for idx in top_indices:
            results.append({
                'question': self.questions[idx],
                'answer': self.answers[idx],
                'score': similarities[idx],
                'index': idx
            })
        
        return results
    
    def get_response(self, query, confidence_threshold=0.0):
        """Get response for a query with optional confidence threshold"""
        results = self.search(query, top_k=1)
        
        if results:
            top_result = results[0]
            if top_result['score'] >= confidence_threshold:
                return top_result['answer'], top_result['score'], top_result['question']
        
        return "I'm sorry, I couldn't find a suitable answer to your question.", 0.0, None


# ============================================================================
# 3. EVALUATION METRICS
# ============================================================================

class QAEvaluator:
    """Evaluate QA system using Exact Match, F1 Score, and BLEU Score"""
    
    @staticmethod
    def exact_match(prediction, reference):
        """
        Exact Match Score: 1 if prediction exactly matches reference, 0 otherwise
        """
        # Normalize both strings
        pred_norm = ' '.join(prediction.lower().split())
        ref_norm = ' '.join(reference.lower().split())
        return 1.0 if pred_norm == ref_norm else 0.0
    
    @staticmethod
    def normalize_answer(answer):
        """Normalize answer for evaluation"""
        # Remove articles
        answer = re.sub(r'\b(a|an|the)\b', ' ', answer)
        # Remove punctuation
        answer = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@\[\]\\^`{|}~]', ' ', answer)
        # Remove extra whitespace
        answer = ' '.join(answer.split())
        return answer.lower()
    
    @staticmethod
    def f1_score(prediction, reference):
        """
        F1 Score: harmonic mean of precision and recall based on token overlap
        """
        pred_tokens = set(QAEvaluator.normalize_answer(prediction).split())
        ref_tokens = set(QAEvaluator.normalize_answer(reference).split())
        
        if len(pred_tokens) == 0 and len(ref_tokens) == 0:
            return 1.0
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0
        
        common = len(pred_tokens & ref_tokens)
        
        if common == 0:
            return 0.0
        
        precision = common / len(pred_tokens)
        recall = common / len(ref_tokens)
        f1 = 2 * (precision * recall) / (precision + recall)
        
        return f1
    
    @staticmethod
    def bleu_score(prediction, reference, weights=(0.5, 0.5)):
        """
        BLEU Score: measures similarity between prediction and reference
        Uses 1-gram and 2-gram (bigram) weights
        """
        pred_tokens = QAEvaluator.normalize_answer(prediction).split()
        ref_tokens = QAEvaluator.normalize_answer(reference).split()
        
        if len(pred_tokens) == 0:
            return 0.0
        
        # Create reference list for BLEU calculation
        reference_list = [ref_tokens]
        
        # Use smoothing function to avoid zero scores
        smoothing_function = SmoothingFunction().method1
        
        # Calculate BLEU score with weights for 1-gram and 2-gram
        bleu = sentence_bleu(
            reference_list,
            pred_tokens,
            weights=weights,
            smoothing_function=smoothing_function
        )
        
        return bleu
    
    @staticmethod
    def evaluate_batch(predictions, references):
        """
        Evaluate a batch of predictions against references
        Returns individual scores and averages
        """
        results = {
            'exact_match': [],
            'f1_score': [],
            'bleu_score': [],
        }
        
        for pred, ref in zip(predictions, references):
            results['exact_match'].append(QAEvaluator.exact_match(pred, ref))
            results['f1_score'].append(QAEvaluator.f1_score(pred, ref))
            results['bleu_score'].append(QAEvaluator.bleu_score(pred, ref))
        
        # Calculate averages
        summary = {
            'exact_match_avg': np.mean(results['exact_match']),
            'f1_score_avg': np.mean(results['f1_score']),
            'bleu_score_avg': np.mean(results['bleu_score']),
            'exact_match_scores': results['exact_match'],
            'f1_scores': results['f1_score'],
            'bleu_scores': results['bleu_score'],
        }
        
        return summary


# ============================================================================
# 4. QA CHATBOT
# ============================================================================

class QAChatbot:
    """Main QA Chatbot using similarity search"""
    
    def __init__(self, qa_pairs):
        """Initialize chatbot with Q&A pairs"""
        self.qa_pairs = qa_pairs
        self.search_engine = SimilaritySearchEngine(qa_pairs)
        self.evaluator = QAEvaluator()
        self.conversation_history = []
    
    def answer(self, query, confidence_threshold=0.0):
        """Answer a user query"""
        response, score, matched_question = self.search_engine.get_response(
            query, confidence_threshold
        )
        
        # Store in conversation history
        self.conversation_history.append({
            'query': query,
            'response': response,
            'score': score,
            'matched_question': matched_question
        })
        
        return response, score, matched_question
    
    def evaluate_on_test_set(self, test_questions, test_answers):
        """Evaluate chatbot on a test set"""
        predictions = []
        scores = []
        
        print("\nEvaluating on test set...")
        print("-" * 80)
        
        for question, reference_answer in zip(test_questions, test_answers):
            response, score, _ = self.answer(question)
            predictions.append(response)
            scores.append(score)
        
        # Calculate evaluation metrics
        evaluation_summary = self.evaluator.evaluate_batch(predictions, test_answers)
        
        return evaluation_summary
    
    def interactive_chat(self):
        """Start interactive chat session"""
        print("\n" + "=" * 80)
        print("QA CHATBOT - Interactive Mode")
        print("=" * 80)
        print("Type 'exit' to quit, 'history' to see conversation history\n")
        
        while True:
            query = input("You: ").strip()
            
            if query.lower() == 'exit':
                print("Chatbot: Goodbye!")
                break
            elif query.lower() == 'history':
                self._print_history()
                continue
            elif not query:
                continue
            
            response, score, matched_question = self.answer(query)
            print(f"\nChatbot: {response}")
            print(f"(Confidence: {score:.4f})")
            if matched_question:
                print(f"(Matched Question: {matched_question})\n")
    
    def _print_history(self):
        """Print conversation history"""
        if not self.conversation_history:
            print("No conversation history yet.\n")
            return
        
        print("\nConversation History:")
        print("-" * 80)
        for i, entry in enumerate(self.conversation_history, 1):
            print(f"{i}. Query: {entry['query']}")
            print(f"   Response: {entry['response']}")
            print(f"   Confidence: {entry['score']:.4f}\n")


# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================

def main():
    """Main function"""
    print("=" * 80)
    print("PRACTICAL 5: Question Answering Chatbot via Similarity Search")
    print("=" * 80)
    
    # Step 1: Load Q&A dataset
    print("\n[Step 1] Loading Q&A Dataset...")
    print("-" * 80)
    loader = QADatasetLoader()
    qa_pairs = loader.create_sample_dataset(size=100)
    print(f"Loaded {len(qa_pairs)} Q&A pairs")
    print("\nSample Q&A pairs:")
    for i, pair in enumerate(qa_pairs[:5], 1):
        print(f"\n{i}. Q: {pair['question']}")
        print(f"   A: {pair['answer']}")
    
    # Step 2: Initialize chatbot
    print("\n\n[Step 2] Initializing QA Chatbot...")
    print("-" * 80)
    chatbot = QAChatbot(qa_pairs)
    print("✓ Chatbot initialized with TF-IDF similarity search")
    
    # Step 3: Test with sample queries
    print("\n\n[Step 3] Testing Chatbot with Sample Queries...")
    print("-" * 80)
    
    test_queries = [
        "What is machine learning?",
        "How do neural networks function?",
        "Tell me about natural language processing",
        "What is BERT used for?",
        "Explain transfer learning",
    ]
    
    print(f"\nRunning {len(test_queries)} test queries:\n")
    
    for i, query in enumerate(test_queries, 1):
        response, score, matched_question = chatbot.answer(query)
        print(f"{i}. Query: {query}")
        print(f"   Answer: {response}")
        print(f"   Similarity Score: {score:.4f}\n")
    
    # Step 4: Evaluation metrics demonstration
    print("\n[Step 4] Demonstrating Evaluation Metrics...")
    print("-" * 80)
    
    evaluator = QAEvaluator()
    
    # Sample prediction and reference
    prediction = "Machine learning is a subset of artificial intelligence"
    reference = "Machine learning is a subset of AI that enables computers to learn from data"
    
    print(f"\nPrediction: {prediction}")
    print(f"Reference: {reference}\n")
    
    em_score = evaluator.exact_match(prediction, reference)
    f1_score = evaluator.f1_score(prediction, reference)
    bleu_score = evaluator.bleu_score(prediction, reference)
    
    print(f"Exact Match Score: {em_score:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"BLEU Score: {bleu_score:.4f}")
    
    # Step 5: Batch evaluation on test set
    print("\n\n[Step 5] Batch Evaluation on Test Set...")
    print("-" * 80)
    
    # Create test set from the dataset (use subset as test)
    test_size = min(10, len(qa_pairs) // 2)
    test_set = qa_pairs[-test_size:]
    
    test_questions = [qa['question'] for qa in test_set]
    test_answers = [qa['answer'] for qa in test_set]
    
    evaluation_summary = chatbot.evaluate_on_test_set(test_questions, test_answers)
    
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Exact Match Score (avg): {evaluation_summary['exact_match_avg']:.4f}")
    print(f"F1 Score (avg): {evaluation_summary['f1_score_avg']:.4f}")
    print(f"BLEU Score (avg): {evaluation_summary['bleu_score_avg']:.4f}")
    
    print("\nDetailed Results:")
    for i, (query, answer) in enumerate(zip(test_questions, test_answers), 1):
        print(f"\n{i}. Query: {query}")
        print(f"   Reference: {answer}")
        print(f"   Exact Match: {evaluation_summary['exact_match_scores'][i-1]:.4f}")
        print(f"   F1 Score: {evaluation_summary['f1_scores'][i-1]:.4f}")
        print(f"   BLEU Score: {evaluation_summary['bleu_scores'][i-1]:.4f}")
    
    # Step 6: Interactive mode option
    print("\n\n[Step 6] Interactive Chat Session")
    print("-" * 80)
    print("\nWould you like to start an interactive chat session? (yes/no)")
    
    user_input = input().strip().lower()
    if user_input in ['yes', 'y']:
        chatbot.interactive_chat()
    
    print("\n" + "=" * 80)
    print("PRACTICAL 5 COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
