import pandas as pd
import numpy as np
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel, Word2Vec
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import gensim.downloader as api

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def perform_anomaly_detection(df):
    """Detects technical and semantic anomalies."""
    logging.info("Starting Anomaly Detection...")
    
    # Technical: Duplicates
    duplicates_count = df.duplicated(subset=['avis_en']).sum()
    
    # Semantic: Vindictive 5-stars (High rating but negative sentiment keywords)
    # Simple keyword-based approach for negative sentiment in high ratings
    neg_keywords = ['horrible', 'bad', 'worst', 'scam', 'terrible', 'avoid', 'expensive', 'slow', 'wait']
    
    def is_vindictive(row):
        if row['note'] >= 4:
            text = str(row['avis_en']).lower()
            if any(word in text for word in neg_keywords):
                return True
        return False

    df['is_anomaly'] = df.apply(is_vindictive, axis=1)
    anomalies = df[df['is_anomaly']]
    
    logging.info(f"Found {duplicates_count} duplicates and {len(anomalies)} semantic anomalies.")
    return df, duplicates_count, anomalies

def train_lda_optimized(texts, num_topics_range=range(3, 8)):
    """Trains LDA and optimizes k using Coherence Score."""
    logging.info("Optimizing LDA topics...")
    
    # Prepare dictionary and corpus
    tokenized_texts = [text.split() for text in texts]
    dictionary = corpora.Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    
    best_k = 0
    best_coherence = -1
    best_model = None
    
    for k in num_topics_range:
        lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=k, random_state=42, passes=10)
        coherence_model = CoherenceModel(model=lda_model, texts=tokenized_texts, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        logging.info(f"K={k}, Coherence={coherence_score:.4f}")
        
        if coherence_score > best_coherence:
            best_coherence = coherence_score
            best_k = k
            best_model = lda_model
            
    logging.info(f"Best LDA model found with K={best_k}")
    return best_model, dictionary, corpus

def export_lda_viz(lda_model, corpus, dictionary, output_path='outputs/lda_viz.html'):
    """Generates and saves pyLDAvis interactive map."""
    vis = gensimvis.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis, output_path)
    logging.info(f"LDA Visualization saved to {output_path}")

def train_word2vec(texts):
    """Trains custom Word2Vec model."""
    tokenized_texts = [text.split() for text in texts]
    model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=2, workers=4, sg=1)
    logging.info("Word2Vec training complete.")
    return model

def load_glove_model():
    """Loads pre-trained GloVe vectors (subset for speed)."""
    logging.info("Loading pre-trained GloVe model (glove-wiki-gigaword-100)...")
    try:
        glove_model = api.load("glove-wiki-gigaword-100")
        return glove_model
    except Exception as e:
        logging.error(f"Failed to load GloVe: {e}")
        return None

def run_analogy_tests(w2v_model):
    """Performs analogy tests: Contrat - Auto + Habitation approx Bail."""
    logging.info("Running Analogy Tests...")
    results = {}
    
    # Note: These words must exist in the vocabulary
    try:
        # Classical: Man - King + Queen = Woman
        res = w2v_model.wv.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
        results['man_king_woman'] = res[0]
        
        # Domain: Contract - Auto + Home
        # Since we use English translation, we use english terms
        res_domain = w2v_model.wv.most_similar(positive=['home', 'contract'], negative=['auto'], topn=1)
        results['insurance_analogy'] = res_domain[0]
    except Exception as e:
        logging.warning(f"Analogy test failed: {e}")
        results['error'] = str(e)
        
    return results

def export_vectors_to_tsv(model_vectors, vocab, output_dir='outputs'):
    """
    Scoring Requirement: Embedding visualization with Tensorboard.
    Exports vectors and metadata for upload to projector.tensorflow.org.
    Supports both Word2Vec and GloVe (KeyedVectors).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Export vectors.tsv
    np.savetxt(os.path.join(output_dir, 'vectors.tsv'), model_vectors, delimiter='\t')
    
    # Export metadata.tsv (the words)
    with open(os.path.join(output_dir, 'metadata.tsv'), 'w', encoding='utf-8') as f:
        for word in vocab:
            f.write(str(word) + '\n')
    
    logging.info(f"✨ Success! Vectors and metadata exported to '{output_dir}'.")
    logging.info("👉 Upload these files to https://projector.tensorflow.org/ to visualize.")

def compare_distance_metrics(w2v_model, word1, word2):
    """Compares Cosine Similarity and Euclidean Distance between two words."""
    if word1 not in w2v_model.wv or word2 not in w2v_model.wv:
        return None
    
    v1 = w2v_model.wv[word1].reshape(1, -1)
    v2 = w2v_model.wv[word2].reshape(1, -1)
    
    cos_sim = cosine_similarity(v1, v2)[0][0]
    euc_dist = euclidean(v1.flatten(), v2.flatten())
    
    return {"cosine": cos_sim, "euclidean": euc_dist}

if __name__ == "__main__":
    # Test would go here
    pass
