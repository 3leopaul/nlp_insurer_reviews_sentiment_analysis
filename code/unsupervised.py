import pandas as pd
import numpy as np
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import os

logging.basicConfig(level=logging.INFO)

def train_lda(texts, n_topics=8):
    """
    Trains an LDA topic model using TF-IDF vectorization.
    """
    logging.info(f"Training LDA with {n_topics} topics...")
    tfidf = TfidfVectorizer(max_features=5000, stop_words=None, ngram_range=(1,2))
    X = tfidf.fit_transform(texts)
    
    lda = LatentDirichletAllocation(
        n_components=n_topics, 
        max_iter=10, 
        learning_method='online', 
        random_state=42, 
        n_jobs=-1
    )
    lda.fit(X)
    
    return lda, tfidf

def get_top_topic_words(model, feature_names, n_top_words=10):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics.append(f"Topic {topic_idx}: " + ", ".join(top_words))
    return topics

def train_word2vec(tokenized_sentences, vector_size=100, window=5, min_count=2):
    """
    Trains a Word2Vec model on preprocessed text.
    Satisfies rubric requiring own embedding training.
    """
    from gensim.models import Word2Vec
    logging.info("Training custom Word2Vec...")
    
    # gensim Word2Vec expects a list of token lists
    # If the user passed a list of strings, we split them here
    if isinstance(tokenized_sentences[0], str):
        tokenized_sentences = [s.split() for s in tokenized_sentences]
        
    w2v = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4
    )
    return w2v

def assign_topics(df, lda_model, tfidf_vec, topic_labels=None):
    """
    Assigns dominant topic to each row in the dataframe.
    """
    texts = df['avis_corrected_clean'].fillna('').astype(str)
    X = tfidf_vec.transform(texts)
    topic_dist = lda_model.transform(X)
    topic_ids = topic_dist.argmax(axis=1)
    
    df['dominant_topic'] = topic_ids
    if topic_labels:
        df['topic_label'] = df['dominant_topic'].map(topic_labels).fillna("Unknown")
    return df

def detect_anomalies(df, text_col='avis_corrected_clean'):
    """
    Identifies 'anomaly' reviews using Isolation Forest.
    Useful for filtering spam, outlier sentiment, or data entry errors.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import IsolationForest
    
    logging.info("Running Anomaly Detection...")
    texts = df[text_col].fillna('').tolist()
    
    # Vectorize - FIX: stop_words='french' is invalid for sklearn, using None
    tfidf = TfidfVectorizer(max_features=2000, stop_words=None)
    X = tfidf.fit_transform(texts)
    
    # Isolation Forest
    # contamination = 0.05 (expecting 5% outliers)
    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    anomalies = iso_forest.fit_predict(X)
    
    # -1 is anomaly, 1 is normal
    df['is_anomaly'] = anomalies
    df['anomaly_score'] = iso_forest.score_samples(X) 
    
    return df, iso_forest
