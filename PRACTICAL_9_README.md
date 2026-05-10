# PRACTICAL 9: Automated Resume Reranking System

## Overview
This practical implements an **Automated Resume Reranking System** that utilizes Natural Language Processing (NLP) and Sentence Similarity techniques to automatically rank candidates based on their resumes' relevance to a specific Job Description.

## System Architecture

### Core Components

#### 1. **ResumeReranker Class**
The main class that orchestrates the entire reranking system:
- **Initialization**: Loads pre-trained sentence embedding model (Sentence Transformers)
- **Text Processing**: Parses and cleans both job descriptions and resumes
- **Feature Extraction**: Extracts skills and keywords from text
- **Similarity Calculation**: Computes cosine similarity between documents
- **Ranking & Analytics**: Generates rankings and dashboard analytics

#### 2. **Key Techniques**

##### **Sentence Embeddings (Semantic Vectorization)**
- Uses `sentence-transformers` library with pre-trained model
- Converts job descriptions and resumes into numerical vectors
- Enables semantic similarity comparison beyond keyword matching

##### **Cosine Similarity**
- Measures the angle between resume and JD vectors
- Scores range from 0 (completely different) to 1 (identical)
- Formula: $\cos(\theta) = \frac{\vec{A} \cdot \vec{B}}{||\vec{A}|| \cdot ||\vec{B}||}$

##### **Multi-Factor Scoring**
Combines three metrics for comprehensive ranking:
1. **Semantic Similarity** (50% weight): Overall relevance based on embeddings
2. **Skill Match Score** (30% weight): Overlap between JD and resume skills
3. **Keyword Match Score** (20% weight): Frequency matching of important terms

Final Score = $0.5 \times \text{Semantic} + 0.3 \times \text{Skills} + 0.2 \times \text{Keywords}$

### Functional Modules

#### **Text Processing**
```python
parse_text(text)          # Clean and normalize text
extract_skills(text)      # Extract technical skills using regex patterns
extract_keywords(text)    # Get top-N most frequent non-stopwords
```

#### **Embedding Generation**
```python
generate_jd_embedding()   # Create embeddings for job description
add_resume()             # Add and embed individual resumes
```

#### **Scoring and Ranking**
```python
calculate_similarity()    # Cosine similarity between vectors
calculate_skill_match()   # Percentage of JD skills found in resume
calculate_keyword_match() # Percentage of JD keywords found in resume
rerank_resumes()         # Generate final rankings
```

#### **Analytics**
```python
generate_analytics()      # Comprehensive recruiter dashboard metrics
visualize_results()       # Generate 4-panel analytics dashboard
```

## Features

### ✨ Input Processing
- **Job Description Parsing**: Extracts structure, requirements, and desired skills
- **Resume Parsing**: Handles various resume formats and layouts
- **Text Normalization**: Removes special characters, standardizes case, handles whitespace

### 🔍 Skill & Keyword Extraction
- **Technical Skills Recognition**: Identifies 40+ common technical skills (Python, React, AWS, etc.)
- **Keyword Extraction**: Extracts top frequent terms excluding stopwords
- **Frequency Analysis**: Counts skill occurrences across candidate pool

### 📊 Semantic Similarity
- **Pre-trained Models**: Uses sentence-transformers (MiniLM-L6-v2)
- **Vector Space**: 384-dimensional embeddings
- **Similarity Scoring**: Cosine similarity on normalized vectors

### 🎯 Multi-Factor Reranking
- **Composite Scoring**: Combines semantic + skill + keyword matching
- **Weighted Combinations**: Customizable scoring weights
- **Comprehensive Results**: 
  - Final rank and score
  - Individual component scores
  - Matching vs. missing skills
  - Direct comparison with JD requirements

### 📈 Recruiter Dashboard Analytics
- **Top Candidates**: Identifies top-3 most relevant resumes
- **Score Distribution**: 
  - High performers (>0.7)
  - Medium performers (0.5-0.7)
  - Low performers (≤0.5)
- **Average Metrics**: Mean scores across all dimensions
- **Skill Gap Analysis**: Identifies commonly missing skills
- **Top Matching Skills**: Shows most prevalent matched skills

### 📉 Visualizations
Four-panel analytics dashboard:
1. **Top 10 Candidates Bar Chart**: Final scores ranked
2. **Score Components Breakdown**: Semantic/Skill/Keyword scores for top 5
3. **Score Distribution Histogram**: Overall score distribution
4. **Category Pie Chart**: High/Medium/Low score distribution

### 💾 Export Capabilities
- **CSV Export**: Ranking results with all metrics
- **JSON Export**: Analytics summary for dashboard integration

## Usage

### Basic Example
```python
from practical_9 import ResumeReranker

# Initialize reranker
reranker = ResumeReranker(model_name='all-MiniLM-L6-v2')

# Generate JD embedding
job_description = "Senior Python Developer with AWS experience..."
reranker.generate_jd_embedding(job_description)

# Add resumes
reranker.add_resume('RES001', resume_text_1, 'Candidate A')
reranker.add_resume('RES002', resume_text_2, 'Candidate B')

# Rerank candidates
results = reranker.rerank_resumes()

# Generate analytics
analytics = reranker.generate_analytics()

# Visualize results
reranker.visualize_results()
```

### Customizing Weights
```python
weights = {
    'semantic_similarity': 0.6,
    'skill_match': 0.25,
    'keyword_match': 0.15
}
results = reranker.rerank_resumes(weights=weights)
```

## Sample Data
The system includes sample data with:
- 1 Job Description: Senior Full Stack Engineer
- 5 Sample Resumes: Ranging from junior to senior levels

Results demonstrate:
- Top candidate: Emma Davis (score ~0.85)
- Mid-tier candidates: Alice Johnson, Carol Williams
- Lower-tier candidates: Bob Smith, David Brown

## Output Files

1. **resume_ranking_results.csv**
   - Columns: rank, resume_id, candidate_name, final_score, semantic_similarity, skill_match_score, keyword_match_score
   - Ready for database import or further analysis

2. **resume_ranking_analytics.json**
   - Structured data for dashboard integration
   - Top 3 candidates, average metrics, score distribution

3. **resume_ranking_analytics.png**
   - 4-panel visualization dashboard
   - High resolution (300 DPI) for presentations

## Technical Stack

### Libraries
- **sentence-transformers**: Pre-trained semantic embeddings
- **scikit-learn**: Cosine similarity calculations
- **pandas & numpy**: Data manipulation
- **nltk**: Tokenization and stopwords
- **matplotlib & seaborn**: Visualizations

### Model
- **Sentence Transformer**: `all-MiniLM-L6-v2`
  - 22M parameters
  - 384-dimensional embeddings
  - ~25x faster than base BERT
  - Trained on 215M+ sentence pairs

## Algorithm Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Parse text | O(n) | O(n) |
| Generate embedding | O(1)* | O(1)* |
| Calculate similarity | O(1)* | O(1)* |
| Rerank N resumes | O(N log N) | O(N) |
| Generate analytics | O(N) | O(N) |

*Constant with respect to number of resumes (depends on embedding model)

## Real-World Applications

1. **HR Automation**: Screen hundreds of resumes for specific job openings
2. **Talent Acquisition**: Identify best-fit candidates automatically
3. **Job Matching Platforms**: Match candidates to suitable job postings
4. **Internal Mobility**: Find employees suitable for new roles
5. **Recruitment Analytics**: Understand candidate pool skill distribution

## Future Enhancements

- Integration with ATS systems
- Multi-language support
- Custom skill ontology
- Historical hiring data integration
- Bias detection and mitigation
- Real-time resume upload and processing
- Batch processing for large resume databases
- Advanced NLP: Named Entity Recognition (NER), Aspect-based sentiment analysis

## Notes

- The system is modular and can be easily extended
- Weights in scoring can be tuned based on domain requirements
- Sentence embeddings capture semantic meaning better than keyword matching alone
- For production, consider using domain-specific embedding models fine-tuned on job descriptions
- The system can handle 1000s of resumes efficiently

## References

- Sentence Transformers: https://www.sbert.net/
- Cosine Similarity: https://en.wikipedia.org/wiki/Cosine_similarity
- NLP Techniques: https://github.com/facebookresearch/fastText
