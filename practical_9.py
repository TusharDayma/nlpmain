"""
Practical 9: Automated Resume Reranking System using NLP and Sentence Similarity
This system reranks candidates based on their resumes' relevance to a specific Job Description
using semantic similarity techniques and natural language processing.
"""

import pandas as pd
import numpy as np
import re
import json
import warnings
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import download as nltk_download
import nltk

warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk_download('punkt', quiet=True)
    nltk_download('stopwords', quiet=True)
    nltk_download('averaged_perceptron_tagger', quiet=True)
except:
    pass

# Try to import sentence-transformers, if not available, provide installation instructions
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("\n⚠️  sentence-transformers not installed. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'sentence-transformers'])
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True


class ResumeReranker:
    """Main class for Resume Reranking System"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the Resume Reranker with sentence embedding model
        Args:
            model_name: Pre-trained sentence transformer model
        """
        self.model = SentenceTransformer(model_name)
        self.stop_words = set(stopwords.words('english'))
        self.jd_embedding = None
        self.resumes = []
        self.resume_embeddings = []
        self.similarity_scores = []
        
    def parse_text(self, text):
        """Parse and clean text"""
        if not isinstance(text, str):
            return ""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_skills(self, text):
        """
        Extract technical skills and keywords from text
        Using common technical terms and skill keywords
        """
        text = text.lower()
        
        # Common technical skills and keywords
        skills_keywords = {
            'programming': ['python', 'java', 'c\\+\\+', 'javascript', 'typescript', 'golang', 'rust', 'php', 'ruby', 'swift'],
            'web': ['html', 'css', 'react', 'angular', 'vue', 'node\\.js', 'express', 'django', 'flask', 'fastapi'],
            'database': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra', 'dynamodb'],
            'cloud': ['aws', 'azure', 'gcp', 'kubernetes', 'docker', 'terraform', 'jenkins', 'gitlab', 'github'],
            'data': ['machine learning', 'deep learning', 'nlp', 'data science', 'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch'],
            'tools': ['git', 'linux', 'unix', 'jira', 'confluence', 'slack', 'salesforce', 'tableau', 'power bi']
        }
        
        extracted_skills = []
        for category, keywords in skills_keywords.items():
            for keyword in keywords:
                if re.search(r'\b' + keyword + r'\b', text):
                    extracted_skills.append(keyword.replace('\\', ''))
        
        return list(set(extracted_skills))
    
    def extract_keywords(self, text, top_n=10):
        """
        Extract top keywords from text (excluding stopwords)
        """
        tokens = word_tokenize(self.parse_text(text))
        # Filter out stopwords and short tokens
        keywords = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        # Count frequency and get top N
        keyword_freq = Counter(keywords)
        return keyword_freq.most_common(top_n)
    
    def generate_jd_embedding(self, job_description):
        """
        Generate embedding for Job Description
        Args:
            job_description: Job description text
        """
        parsed_jd = self.parse_text(job_description)
        self.jd_embedding = self.model.encode(parsed_jd, convert_to_numpy=True)
        self.job_description = job_description
        self.jd_keywords = self.extract_keywords(job_description, top_n=15)
        self.jd_skills = self.extract_skills(job_description)
        return self.jd_embedding
    
    def add_resume(self, resume_id, resume_text, candidate_name="Unknown"):
        """
        Add a resume to the system
        Args:
            resume_id: Unique identifier for the resume
            resume_text: Resume content
            candidate_name: Name of the candidate
        """
        parsed_resume = self.parse_text(resume_text)
        embedding = self.model.encode(parsed_resume, convert_to_numpy=True)
        
        resume_data = {
            'resume_id': resume_id,
            'candidate_name': candidate_name,
            'resume_text': resume_text,
            'parsed_text': parsed_resume,
            'embedding': embedding,
            'skills': self.extract_skills(resume_text),
            'keywords': self.extract_keywords(resume_text, top_n=15)
        }
        
        self.resumes.append(resume_data)
        self.resume_embeddings.append(embedding)
    
    def calculate_similarity(self, resume_embedding):
        """Calculate cosine similarity between resume and JD"""
        if self.jd_embedding is None:
            raise ValueError("Job Description embedding not generated. Call generate_jd_embedding() first.")
        
        similarity = cosine_similarity([resume_embedding], [self.jd_embedding])[0][0]
        return similarity
    
    def calculate_skill_match(self, resume_skills):
        """Calculate skill matching score"""
        if not self.jd_skills:
            return 0
        
        matching_skills = set(resume_skills) & set(self.jd_skills)
        skill_match_score = len(matching_skills) / len(self.jd_skills) if self.jd_skills else 0
        return skill_match_score
    
    def calculate_keyword_match(self, resume_keywords):
        """Calculate keyword matching score"""
        jd_keywords_set = set([kw[0] for kw in self.jd_keywords])
        resume_keywords_set = set([kw[0] for kw in resume_keywords])
        
        matching_keywords = jd_keywords_set & resume_keywords_set
        keyword_match_score = len(matching_keywords) / len(jd_keywords_set) if jd_keywords_set else 0
        return keyword_match_score
    
    def rerank_resumes(self, weights=None):
        """
        Rerank all resumes based on multiple factors
        Args:
            weights: Dictionary with weights for different factors
                    {'semantic_similarity': 0.5, 'skill_match': 0.3, 'keyword_match': 0.2}
        """
        if not weights:
            weights = {
                'semantic_similarity': 0.5,
                'skill_match': 0.3,
                'keyword_match': 0.2
            }
        
        results = []
        
        for resume in self.resumes:
            # Calculate different scoring metrics
            semantic_score = self.calculate_similarity(resume['embedding'])
            skill_score = self.calculate_skill_match(resume['skills'])
            keyword_score = self.calculate_keyword_match(resume['keywords'])
            
            # Calculate weighted final score
            final_score = (
                semantic_score * weights['semantic_similarity'] +
                skill_score * weights['skill_match'] +
                keyword_score * weights['keyword_match']
            )
            
            results.append({
                'rank': 0,  # Will be updated
                'resume_id': resume['resume_id'],
                'candidate_name': resume['candidate_name'],
                'semantic_similarity': round(semantic_score, 4),
                'skill_match_score': round(skill_score, 4),
                'keyword_match_score': round(keyword_score, 4),
                'final_score': round(final_score, 4),
                'matching_skills': list(set(resume['skills']) & set(self.jd_skills)),
                'missing_skills': list(set(self.jd_skills) - set(resume['skills']))
            })
        
        # Sort by final score (descending)
        results = sorted(results, key=lambda x: x['final_score'], reverse=True)
        
        # Update ranks
        for idx, result in enumerate(results, 1):
            result['rank'] = idx
        
        self.ranking_results = results
        return results
    
    def generate_analytics(self):
        """Generate analytics for recruiter dashboard"""
        if not hasattr(self, 'ranking_results'):
            raise ValueError("Must call rerank_resumes() first")
        
        results_df = pd.DataFrame(self.ranking_results)
        
        analytics = {
            'total_candidates': len(self.ranking_results),
            'top_3_candidates': self.ranking_results[:3],
            'average_semantic_similarity': round(results_df['semantic_similarity'].mean(), 4),
            'average_skill_match': round(results_df['skill_match_score'].mean(), 4),
            'average_keyword_match': round(results_df['keyword_match_score'].mean(), 4),
            'score_distribution': {
                'high': len(results_df[results_df['final_score'] > 0.7]),
                'medium': len(results_df[(results_df['final_score'] > 0.5) & (results_df['final_score'] <= 0.7)]),
                'low': len(results_df[results_df['final_score'] <= 0.5])
            },
            'skill_gap_analysis': self._analyze_skill_gaps(results_df),
            'top_matching_skills': self._get_top_matching_skills(results_df)
        }
        
        return analytics
    
    def _analyze_skill_gaps(self, results_df):
        """Analyze skill gaps across candidates"""
        all_missing_skills = []
        for missing in results_df['missing_skills']:
            all_missing_skills.extend(missing)
        
        skill_gap_freq = Counter(all_missing_skills)
        return skill_gap_freq.most_common(10)
    
    def _get_top_matching_skills(self, results_df):
        """Get most commonly found matching skills"""
        all_matching_skills = []
        for matching in results_df['matching_skills']:
            all_matching_skills.extend(matching)
        
        skill_freq = Counter(all_matching_skills)
        return skill_freq.most_common(10)
    
    def print_ranking_results(self, top_n=10):
        """Print formatted ranking results"""
        if not hasattr(self, 'ranking_results'):
            raise ValueError("Must call rerank_resumes() first")
        
        print("\n" + "="*100)
        print("RESUME RANKING RESULTS".center(100))
        print("="*100 + "\n")
        
        results = self.ranking_results[:top_n]
        
        for result in results:
            print(f"Rank #{result['rank']}: {result['candidate_name']} (ID: {result['resume_id']})")
            print(f"  Final Score: {result['final_score']:.4f}")
            print(f"  ├─ Semantic Similarity: {result['semantic_similarity']:.4f}")
            print(f"  ├─ Skill Match Score: {result['skill_match_score']:.4f}")
            print(f"  └─ Keyword Match Score: {result['keyword_match_score']:.4f}")
            
            if result['matching_skills']:
                print(f"  ✓ Matching Skills ({len(result['matching_skills'])}): {', '.join(result['matching_skills'][:5])}")
            
            if result['missing_skills']:
                print(f"  ✗ Missing Skills ({len(result['missing_skills'])}): {', '.join(result['missing_skills'][:5])}")
            
            print()
    
    def visualize_results(self, save_path=None):
        """Generate visualizations for the ranking results"""
        if not hasattr(self, 'ranking_results'):
            raise ValueError("Must call rerank_resumes() first")
        
        results_df = pd.DataFrame(self.ranking_results)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Resume Reranking System - Analytics Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Ranking Scores (top 10)
        top_candidates = results_df.head(10)
        ax1 = axes[0, 0]
        bars = ax1.barh(range(len(top_candidates)), top_candidates['final_score'].values)
        ax1.set_yticks(range(len(top_candidates)))
        ax1.set_yticklabels([f"#{r} - {name[:20]}" for r, name in zip(top_candidates['rank'], top_candidates['candidate_name'])])
        ax1.set_xlabel('Final Score')
        ax1.set_title('Top 10 Candidates - Final Scores')
        ax1.invert_yaxis()
        for i, bar in enumerate(bars):
            ax1.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                    f'{bar.get_width():.3f}', ha='left', va='center', fontsize=9)
        
        # 2. Score Components Breakdown (top 5)
        top_5 = results_df.head(5)
        ax2 = axes[0, 1]
        x = np.arange(len(top_5))
        width = 0.25
        ax2.bar(x - width, top_5['semantic_similarity'], width, label='Semantic Similarity')
        ax2.bar(x, top_5['skill_match_score'], width, label='Skill Match')
        ax2.bar(x + width, top_5['keyword_match_score'], width, label='Keyword Match')
        ax2.set_xlabel('Candidate')
        ax2.set_ylabel('Score')
        ax2.set_title('Score Components - Top 5 Candidates')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"#{r}" for r in top_5['rank']], rotation=0)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Score Distribution
        ax3 = axes[1, 0]
        ax3.hist(results_df['final_score'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax3.axvline(results_df['final_score'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {results_df["final_score"].mean():.3f}')
        ax3.set_xlabel('Final Score')
        ax3.set_ylabel('Number of Candidates')
        ax3.set_title('Score Distribution')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Category Distribution
        ax4 = axes[1, 1]
        high = len(results_df[results_df['final_score'] > 0.7])
        medium = len(results_df[(results_df['final_score'] > 0.5) & (results_df['final_score'] <= 0.7)])
        low = len(results_df[results_df['final_score'] <= 0.5])
        
        categories = ['High (>0.7)', 'Medium (0.5-0.7)', 'Low (≤0.5)']
        values = [high, medium, low]
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        
        wedges, texts, autotexts = ax4.pie(values, labels=categories, autopct='%1.1f%%', colors=colors, startangle=90)
        ax4.set_title('Candidate Distribution by Score Category')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Visualization saved to {save_path}")
        else:
            plt.savefig('resume_ranking_analytics.png', dpi=300, bbox_inches='tight')
            print(f"\n✓ Visualization saved to resume_ranking_analytics.png")
        
        plt.close()


def create_sample_data():
    """Create sample resumes and job description for demonstration"""
    
    job_description = """
    Senior Software Engineer - Full Stack Development
    
    We are seeking an experienced Senior Software Engineer with expertise in full-stack development.
    
    Required Skills:
    - 5+ years of software development experience
    - Proficiency in Python, JavaScript/TypeScript
    - Experience with React, Node.js, and Express
    - Strong SQL and database design knowledge (PostgreSQL, MySQL)
    - Experience with AWS, Docker, and Kubernetes
    - Git and CI/CD pipelines (Jenkins, GitLab)
    - Machine Learning basics or NLP experience is a plus
    - Strong problem-solving and system design skills
    
    Responsibilities:
    - Design and develop scalable web applications
    - Collaborate with product and design teams
    - Write clean, maintainable code with comprehensive tests
    - Participate in code reviews and technical discussions
    - Mentor junior developers
    """
    
    sample_resumes = [
        {
            'id': 'RES001',
            'name': 'Alice Johnson',
            'text': """
            Alice Johnson
            Senior Software Engineer
            
            Experience:
            - Lead Software Engineer at TechCorp (2019-Present)
              * Developed full-stack web applications using React, Node.js, and Express
              * Designed microservices architecture using Python and Docker
              * Implemented CI/CD pipelines with Jenkins and GitLab
              * Mentored 3 junior developers
            
            - Software Engineer at StartupXYZ (2016-2019)
              * Built scalable APIs using Python and Flask
              * Worked with PostgreSQL and MySQL databases
              * Deployed applications on AWS using Kubernetes
            
            Skills:
            - Languages: Python, JavaScript, TypeScript
            - Frontend: React, HTML, CSS
            - Backend: Node.js, Express, Flask
            - Database: PostgreSQL, MySQL, Redis
            - Cloud: AWS, Docker, Kubernetes
            - Tools: Git, Jenkins, GitLab
            
            Education:
            - B.S. in Computer Science, State University (2016)
            """
        },
        {
            'id': 'RES002',
            'name': 'Bob Smith',
            'text': """
            Bob Smith
            Software Developer
            
            Experience:
            - Junior Developer at WebAgency (2021-Present)
              * Developed frontend components using React
              * Fixed bugs and implemented features for client projects
            
            - Junior Developer at LocalStartup (2020-2021)
              * Created static websites with HTML and CSS
              * Basic JavaScript functionality
            
            Skills:
            - Languages: JavaScript, HTML, CSS
            - Frontend: React
            - Database: Basic SQL knowledge
            
            Education:
            - Bootcamp Certificate, Coding School (2020)
            """
        },
        {
            'id': 'RES003',
            'name': 'Carol Williams',
            'text': """
            Carol Williams
            Full Stack Engineer
            
            Experience:
            - Software Engineer at CloudServices (2017-Present)
              * 6+ years of full-stack development experience
              * Expert in Python, JavaScript, and TypeScript
              * Architected microservices using Docker and Kubernetes
              * Led database optimization projects using PostgreSQL
              * Experience with machine learning and NLP projects
              * AWS certified and experienced
            
            - Backend Engineer at FinTech Company (2015-2017)
              * Developed robust APIs using Python Django
              * Implemented complex database queries and optimization
            
            Skills:
            - Languages: Python, JavaScript, TypeScript, Go
            - Frontend: React, Angular
            - Backend: Node.js, Express, Django, FastAPI
            - Database: PostgreSQL, MySQL, MongoDB
            - Cloud: AWS, GCP, Azure
            - ML/AI: TensorFlow, Scikit-learn, NLP
            - DevOps: Docker, Kubernetes, CI/CD, Jenkins
            - Tools: Git, GitHub, GitLab
            
            Education:
            - M.S. in Computer Science, Tech University (2015)
            - B.S. in Engineering, Tech University (2013)
            """
        },
        {
            'id': 'RES004',
            'name': 'David Brown',
            'text': """
            David Brown
            Java Developer
            
            Experience:
            - Java Developer at EnterpriseSoft (2018-Present)
              * Developed backend services using Java and Spring
              * Worked with Oracle databases
              * Experience with microservices architecture
            
            Skills:
            - Languages: Java, SQL
            - Backend: Spring, Spring Boot
            - Database: Oracle, MySQL
            - Tools: Git, Maven
            
            Education:
            - B.S. in Computer Science (2018)
            """
        },
        {
            'id': 'RES005',
            'name': 'Emma Davis',
            'text': """
            Emma Davis
            Full Stack Developer
            
            Experience:
            - Senior Developer at TechInnovate (2016-Present)
              * 8+ years of development experience
              * Strong Python and JavaScript expertise
              * Built scalable applications with React and Node.js
              * Extensive Docker and Kubernetes experience
              * AWS architect level experience
              * Team lead and mentoring experience
              * Git and Jenkins CI/CD expert
            
            - Developer at WebStartup (2014-2016)
              * Full-stack web development
              * Database design and optimization
            
            Skills:
            - Languages: Python, JavaScript, TypeScript
            - Frontend: React, Vue.js
            - Backend: Node.js, Express, Flask, Django
            - Database: PostgreSQL, MySQL, MongoDB, Redis
            - Cloud: AWS, Docker, Kubernetes
            - DevOps: Jenkins, GitLab CI, Terraform
            - Tools: Git, GitHub
            
            Education:
            - B.S. in Computer Engineering (2014)
            """
        }
    ]
    
    return job_description, sample_resumes


def main():
    print("\n" + "="*100)
    print("PRACTICAL 9: AUTOMATED RESUME RERANKING SYSTEM".center(100))
    print("Using NLP and Sentence Similarity Techniques".center(100))
    print("="*100 + "\n")
    
    # Initialize the reranker
    print("🔧 Initializing Resume Reranker with Sentence Transformer model...")
    reranker = ResumeReranker(model_name='all-MiniLM-L6-v2')
    print("✓ Reranker initialized successfully.\n")
    
    # Load sample data
    print("📋 Loading sample data...")
    job_description, sample_resumes = create_sample_data()
    print(f"✓ Loaded {len(sample_resumes)} sample resumes\n")
    
    # Display Job Description Analysis
    print("-" * 100)
    print("JOB DESCRIPTION ANALYSIS")
    print("-" * 100)
    
    jd_embedding = reranker.generate_jd_embedding(job_description)
    print(f"\n✓ Job Description parsed and embedded")
    print(f"  Embedding dimension: {len(jd_embedding)}")
    print(f"  Extracted Skills: {reranker.jd_skills}")
    print(f"  Top Keywords: {[kw[0] for kw in reranker.jd_keywords[:10]]}\n")
    
    # Add all resumes
    print("-" * 100)
    print("PROCESSING RESUMES")
    print("-" * 100 + "\n")
    
    for resume in sample_resumes:
        reranker.add_resume(resume['id'], resume['text'], resume['name'])
        print(f"✓ Processed: {resume['name']} ({resume['id']})")
    
    print(f"\n✓ Total resumes processed: {len(reranker.resumes)}\n")
    
    # Rerank resumes
    print("-" * 100)
    print("RERANKING CANDIDATES")
    print("-" * 100 + "\n")
    
    ranking_results = reranker.rerank_resumes()
    
    # Display results
    reranker.print_ranking_results(top_n=len(sample_resumes))
    
    # Generate analytics
    print("-" * 100)
    print("DASHBOARD ANALYTICS")
    print("-" * 100 + "\n")
    
    analytics = reranker.generate_analytics()
    
    print(f"Total Candidates Evaluated: {analytics['total_candidates']}")
    print(f"Average Semantic Similarity: {analytics['average_semantic_similarity']:.4f}")
    print(f"Average Skill Match Score: {analytics['average_skill_match']:.4f}")
    print(f"Average Keyword Match Score: {analytics['average_keyword_match']:.4f}\n")
    
    print("Score Distribution:")
    print(f"  ✓ High Score (>0.7): {analytics['score_distribution']['high']} candidates")
    print(f"  ⚠ Medium Score (0.5-0.7): {analytics['score_distribution']['medium']} candidates")
    print(f"  ✗ Low Score (≤0.5): {analytics['score_distribution']['low']} candidates\n")
    
    print("Top Matching Skills Across All Candidates:")
    for skill, count in analytics['top_matching_skills'][:5]:
        print(f"  • {skill}: found in {count} resumes")
    
    print("\nSkill Gaps (Most Common Missing Skills):")
    for skill, count in analytics['skill_gap_analysis'][:5]:
        print(f"  • {skill}: missing in {count} resumes")
    
    # Generate visualizations
    print("\n" + "-" * 100)
    print("GENERATING VISUALIZATIONS")
    print("-" * 100 + "\n")
    
    reranker.visualize_results()
    
    # Export results to CSV
    print("\n" + "-" * 100)
    print("EXPORTING RESULTS")
    print("-" * 100 + "\n")
    
    results_df = pd.DataFrame(ranking_results)
    results_df = results_df[['rank', 'resume_id', 'candidate_name', 'final_score', 
                              'semantic_similarity', 'skill_match_score', 'keyword_match_score']]
    results_df.to_csv('resume_ranking_results.csv', index=False)
    print("✓ Results exported to resume_ranking_results.csv\n")
    
    # Export analytics to JSON
    # Convert numpy types to Python native types for JSON serialization
    analytics_json = {
        'total_candidates': int(analytics['total_candidates']),
        'average_semantic_similarity': float(analytics['average_semantic_similarity']),
        'average_skill_match': float(analytics['average_skill_match']),
        'average_keyword_match': float(analytics['average_keyword_match']),
        'score_distribution': {k: int(v) for k, v in analytics['score_distribution'].items()},
        'top_3_candidates': [
            {
                'rank': int(c['rank']),
                'candidate_name': str(c['candidate_name']),
                'final_score': float(c['final_score'])
            } for c in analytics['top_3_candidates']
        ]
    }
    
    with open('resume_ranking_analytics.json', 'w') as f:
        json.dump(analytics_json, f, indent=2)
    print("✓ Analytics exported to resume_ranking_analytics.json\n")
    
    print("="*100)
    print("✓ Resume Reranking System Completed Successfully!".center(100))
    print("="*100 + "\n")


if __name__ == "__main__":
    main()
