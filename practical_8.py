import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_data(filepath='spam.csv'):
    # SMS spam dataset often has 'v1' (label) and 'v2' (text), and some empty cols
    df = pd.read_csv(filepath, encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    print("Loaded dataset successfully.")
    return df

def main():
    print("--- Practical 8: SMS/Email Spam Detection ---\n")
    df = load_data()
    
    X = df['text']
    y = df['label']
    
    # Handle imbalanced dataset (Demonstration: print distribution)
    print("Class Distribution:")
    print(y.value_counts(normalize=True))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\nApplying TF-IDF Vectorization...")
    # For demonstration we use TF-IDF, CountVectorizer and Embeddings are conceptually mentioned in the code via comments
    tfidf = TfidfVectorizer(max_features=3000, stop_words='english')
    X_train_tfidf = tfidf.fit_transform(X_train).toarray()
    X_test_tfidf = tfidf.transform(X_test).toarray()
    
    print("\nTraining Models...")
    
    nb_model = MultinomialNB()
    svm_model = SVC(kernel='linear', probability=True, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    ensemble_model = VotingClassifier(
        estimators=[('nb', nb_model), ('svm', svm_model), ('rf', rf_model)],
        voting='soft'
    )
    
    models = {
        'Naive Bayes': nb_model,
        'SVM': svm_model,
        'Random Forest': rf_model,
        'Ensemble Voting': ensemble_model
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"{name} Accuracy: {acc:.4f}")
        # print(classification_report(y_test, y_pred))
        
    plt.figure(figsize=(8, 5))
    plt.bar(results.keys(), results.values(), color=['blue', 'green', 'orange', 'red'])
    plt.title('Spam Detection Model Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    for i, v in enumerate(results.values()):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
    plt.savefig('practical_8_spam_comparison.png')
    print("\nSaved accuracy comparison chart to 'practical_8_spam_comparison.png'")

if __name__ == "__main__":
    main()
