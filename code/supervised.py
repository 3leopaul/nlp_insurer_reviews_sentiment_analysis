import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)

def train_ridge_regression(X_train, X_test, y_train, y_test):
    """
    Trains a Ridge Regression model for star prediction (continuous).
    Excellent for capturing neutral/middle-ground nuances.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error

    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    logging.info("Training TF-IDF + Ridge Regression...")
    model = Ridge(alpha=1.0)
    model.fit(X_train_tfidf, y_train)

    preds = model.predict(X_test_tfidf)
    mae = mean_absolute_error(y_test, preds)
    logging.info(f"Ridge Regression MAE: {mae:.3f} stars")

    return model, tfidf, mae

def train_logistic_regression(X_train, X_test, y_train, y_test):
    """
    Trains a Logistic Regression model for sentiment classification (3 classes).
    Provides high interpretability for SHAP/LIME.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report

    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    logging.info("Training TF-IDF + Logistic Regression...")
    model = LogisticRegression(max_iter=1000, multi_class='multinomial')
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    logging.info("Trainer report:\n" + classification_report(y_test, y_pred))

    return model, tfidf

def run_camembert_inference(texts):
    """
    Runs binary sentiment inference using a pre-trained CamemBERT model.
    """
    from transformers import pipeline
    logging.info("Loading CamemBERT pipeline...")
    # Using a specialized French sentiment model
    classifier = pipeline("sentiment-analysis", model="tblard/tf-allocine", framework="tf")
    results = classifier(texts)
    return [r['label'] for r in results]

def stars_to_sentiment(stars):
    """Threshold logic for ridge regression continuous output."""
    if stars >= 3.75: return "positive"
    if stars <= 2.25: return "negative"
    return "neutral"

def train_bilstm(X_train, X_val, y_train_cat, y_val_cat, vocab_size=20000, max_len=150):
    """
    Trains a Bidirectional LSTM Network for sentiment analysis.
    Architecture based on the notebook's most successful GloVe+BiLSTM setup.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
    
    logging.info("Building Bi-LSTM model...")
    model = Sequential([
        Embedding(vocab_size, 100, input_length=max_len),
        Bidirectional(LSTM(128, return_sequences=True, dropout=0.2)),
        Bidirectional(LSTM(64, dropout=0.2)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(y_train_cat.shape[1], activation='softmax')
    ], name="BiLSTM_Sentiment")
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Fit (fast training for prototype)
    model.fit(X_train, y_train_cat, validation_data=(X_val, y_val_cat), epochs=8, batch_size=32, verbose=0)
    
    return model

def get_model_metrics(y_true, y_pred, labels):
    """Returns classification report and confusion matrix as dict."""
    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    return report, cm
