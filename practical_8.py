import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF, PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import warnings

warnings.filterwarnings('ignore')

def load_data(filepath='News_Category_Dataset_v3.json', sample_size=2000):
    df = pd.read_json(filepath, lines=True)
    # We'll use headline and short_description
    df['text'] = df['headline'] + " " + df['short_description']
    df = df.sample(sample_size, random_state=42)
    print("Loaded dataset successfully.")
    return df['text'].tolist()

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

def main():
    print("--- Practical 9: Topic Modelling & Document Clustering ---\n")
    documents = load_data()
    
    no_features = 1000
    no_topics = 5
    no_top_words = 10
    
    # 1. Topic Modeling (LDA & NMF)
    print("\nApplying TF and TF-IDF...")
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
    tf = tf_vectorizer.fit_transform(documents)
    tf_feature_names = tf_vectorizer.get_feature_names_out()
    
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(documents)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    
    print("\n--- Latent Dirichlet Allocation (LDA) ---")
    lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', random_state=42).fit(tf)
    display_topics(lda, tf_feature_names, no_top_words)
    
    print("\n--- Non-negative Matrix Factorization (NMF) ---")
    nmf = NMF(n_components=no_topics, random_state=42, init='nndsvd').fit(tfidf)
    display_topics(nmf, tfidf_feature_names, no_top_words)
    
    # 2. Document Clustering (K-Means)
    print("\n--- K-Means Clustering ---")
    num_clusters = 5
    km = KMeans(n_clusters=num_clusters, random_state=42)
    km.fit(tfidf)
    clusters = km.labels_
    
    # Dimensionality Reduction for Visualization
    print("\nPerforming PCA and t-SNE for visualization...")
    dense_tfidf = tfidf.todense()
    dense_tfidf = np.asarray(dense_tfidf)
    
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(dense_tfidf)
    
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(dense_tfidf)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='viridis', alpha=0.5)
    plt.title('PCA Document Clustering')
    plt.colorbar(scatter)
    
    plt.subplot(1, 2, 2)
    scatter2 = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=clusters, cmap='viridis', alpha=0.5)
    plt.title('t-SNE Document Clustering')
    plt.colorbar(scatter2)
    
    plt.tight_layout()
    plt.savefig('practical_9_clustering.png')
    print("Saved clustering visualizations to 'practical_9_clustering.png'")

if __name__ == "__main__":
    main()
