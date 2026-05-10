import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional

def load_data(filepath='training.1600000.processed.noemoticon.csv', sample_size=10000):
    # Sentiment140 dataset has no header, columns: target, ids, date, flag, user, text
    df = pd.read_csv(filepath, encoding='latin-1', header=None, 
                     names=['target', 'ids', 'date', 'flag', 'user', 'text'])
    # target: 0 = negative, 2 = neutral, 4 = positive
    df = df[['target', 'text']].sample(sample_size, random_state=42)
    df['target'] = df['target'].replace(4, 1) # Map 4 to 1 for binary classification
    print("Loaded dataset successfully.")
    return df

def plot_history(history):
    # Plot training & validation accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('practical_3_learning_curves.png')
    print("Saved learning curves to 'practical_3_learning_curves.png'")

def plot_roc_curve(y_test, y_pred_prob):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('practical_3_roc_curve.png')
    print("Saved ROC curve to 'practical_3_roc_curve.png'")

def main():
    print("--- Practical 3: Sentiment Classification using LSTM ---\n")
    df = load_data()
    
    texts = df['text'].astype(str).values
    labels = df['target'].values
    
    # Preprocessing, Tokenization, Padding
    max_words = 10000
    max_len = 50
    
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)
    
    # Model Building
    print("Building BiLSTM Model...")
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=64, input_length=max_len),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    
    print("\nTraining Model...")
    # Using small epochs for demonstration
    history = model.fit(X_train, y_train, epochs=3, batch_size=32, validation_split=0.2, verbose=1)
    
    print("\nEvaluating Model...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    y_pred_prob = model.predict(X_test)
    
    plot_history(history)
    plot_roc_curve(y_test, y_pred_prob)

if __name__ == "__main__":
    main()
