import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, classification_report
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_modeling_data(train_df, test_df=None, text_col='avis_corrected_clean', label_col='sentiment', test_size=0.2):
    """Splits data and prepares labels for sentiment analysis."""
    if test_df is not None:
        X_train = train_df[text_col].astype(str).values
        y_train = train_df[label_col].values
        X_test = test_df[text_col].astype(str).values
        y_test = test_df[label_col].values
        return X_train, X_test, y_train, y_test
    
    # Stratified split using the label column
    X = train_df[text_col].astype(str).values
    y = train_df[label_col].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

def train_ridge_regression(X_train, X_test, y_train_stars, y_test_stars):
    """
    Trains Ridge Regression on TF-IDF features to predict star continuous ratings.
    Notebook identified this as the best model for neutral (3-star) sentiment detection.
    """
    logging.info("Training TF-IDF + Ridge Regression...")
    tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_tfidf, y_train_stars)
    
    y_pred_stars = ridge.predict(X_test_tfidf)
    mae = mean_absolute_error(y_test_stars, y_pred_stars)
    logging.info(f"Ridge Regression MAE: {mae:.3f} stars")
    
    return ridge, tfidf, mae

def stars_to_sentiment(stars, low=2.5, high=3.5):
    """Maps continuous star prediction to sentiment class."""
    if stars >= high: return 'positive'
    elif stars >= low: return 'neutral'
    else: return 'negative'

def train_logistic_regression(X_train, X_test, y_train, y_test):
    """
    Trains TF-IDF + Logistic Regression. 
    Notebook uses this specifically as the best classical ML logic for LIME & SHAP explainability.
    """
    logging.info("Training TF-IDF + Logistic Regression...")
    tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    lr_model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr_model.fit(X_train_tfidf, y_train)
    
    preds = lr_model.predict(X_test_tfidf)
    report = classification_report(y_test, preds)
    logging.info(f"Trainer report:\n{report}")
    
    return lr_model, tfidf

def run_camembert_inference(texts):
    """
    Runs CamemBERT binary classification (from huggingface tblard/tf-allocine).
    Identified by the notebook as the most accurate model for positive/negative discrimination.
    """
    from transformers import pipeline, CamembertTokenizer
    BERT_MODEL = 'tblard/tf-allocine'
    logging.info(f'Loading {BERT_MODEL}...')
    bert_tokenizer = CamembertTokenizer.from_pretrained(BERT_MODEL)
    
    bert_clf = pipeline(
        'text-classification',
        model=BERT_MODEL,
        tokenizer=bert_tokenizer,
        truncation=True,
        max_length=512
    )
    
    # Process
    results = bert_clf([str(t) for t in texts])
    preds = []
    for r in results:
        label = r['label']
        if 'NEGATIVE' in label.upper() or label == 'LABEL_0':
            preds.append('negative')
        else:
            preds.append('positive')
    return preds
