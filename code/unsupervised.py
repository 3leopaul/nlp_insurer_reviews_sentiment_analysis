import pandas as pd
import numpy as np
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel, Word2Vec
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import matplotlib.pyplot as plt
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_lda(texts, num_topics=8):
    """Trains LDA model based on the notebook's optimal K=8."""
    logging.info("Training LDA model...")
    tokenized_texts = [str(text).split() for text in texts if isinstance(text, str)]
    dictionary = corpora.Dictionary(tokenized_texts)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42, passes=10)
    return lda_model, dictionary, corpus

def export_lda_viz(lda_model, corpus, dictionary, output_path='outputs/lda_viz.html'):
    """Generates and saves pyLDAvis interactive map."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    vis = gensimvis.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis, output_path)
    logging.info(f"LDA Visualization saved to {output_path}")

def train_word2vec(texts):
    """Trains custom Word2Vec model on the French texts."""
    tokenized_texts = [str(text).split() for text in texts if isinstance(text, str) and len(str(text).split()) >= 3]
    model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=5, workers=4, sg=1)
    return model
    
def get_dominant_topic(lda_model, bow):
    """Assigns the dominant topic to a document."""
    if not bow: return -1
    topic_probs = lda_model.get_document_topics(bow, minimum_probability=0)
    return max(topic_probs, key=lambda x: x[1])[0]

def attach_topics(df, lda_model, dictionary):
    """Attaches dominant LDA topic for each review in the DataFrame."""
    texts = df['avis_corrected_clean'].tolist()
    topic_ids = []
    
    # Notebook's mapping
    topic_labels = {
        0: 'Sinistres Auto', 
        1: 'Augmentation des Prix',
        2: 'Satisfaction Générale', 
        3: 'Service Client Négatif',
        4: 'Mutuelle Santé', 
        5: 'Délais Administratifs',
        6: 'Prévoyance & Vie', 
        7: 'Accueil Téléphonique Positif'
    }

    for text in texts:
        if isinstance(text, str):
            bow = dictionary.doc2bow(text.split())
            topic_ids.append(get_dominant_topic(lda_model, bow))
        else:
            topic_ids.append(-1)
            
    df['dominant_topic'] = topic_ids
    df['topic_label'] = df['dominant_topic'].map(topic_labels).fillna("Unknown")
    return df
