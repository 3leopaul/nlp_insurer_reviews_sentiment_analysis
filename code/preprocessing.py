import pandas as pd
import numpy as np
import re
import os
import glob
import logging
import spacy
from nltk.corpus import stopwords
from deep_translator import GoogleTranslator
from spellchecker import SpellChecker

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load spacy model
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    logging.error("SpaCy model 'en_core_web_sm' not found. Please run 'python3 -m spacy download en_core_web_sm'")
    raise

def load_all_data(data_dir):
    """Loads all 35 .xlsx files and cleans column names."""
    files = glob.glob(os.path.join(data_dir, "*.xlsx"))
    dfs = []
    for f in files:
        df = pd.read_excel(f)
        dfs.append(df)
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Normalize column names
    full_df.columns = (
        full_df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[\s/]+", "_", regex=True)
    )
    logging.info(f"Loaded {len(full_df)} rows. Columns normalized: {full_df.columns.tolist()}")
    return full_df

def recover_placeholders(df):
    """
    Notebook 'Brain': Uses deep_translator to fix rows where avis_en is 'Loading...' or empty.
    """
    placeholders = ["loading...", "loading", "nan", ""]
    mask = (
        df['avis_en'].isna() | 
        df['avis_en'].astype(str).str.strip().str.lower().isin(placeholders)
    )
    
    needed = mask.sum()
    if needed > 0:
        logging.info(f"Recovering {needed} placeholder translations using GoogleTranslator...")
        translator = GoogleTranslator(source='fr', target='en')
        
        def safe_translate(text):
            try:
                if not isinstance(text, str) or not text.strip(): return ""
                return translator.translate(text[:4500])
            except Exception: return ""
            
        df.loc[mask, 'avis_en'] = df.loc[mask, 'avis'].apply(safe_translate)
    return df

def correct_spelling(text, domain_exclusions=None):
    """Corrects spelling while ignoring domain-specific proper nouns."""
    if not isinstance(text, str) or len(text) < 3: return text
    
    spell = SpellChecker(language='en')
    if domain_exclusions:
        spell.word_frequency.load_words(domain_exclusions)
    
    # TextBlob logic from notebook was precision-focused; 
    # here we use SpellChecker but with faster tokenization
    words = text.split()
    misspelled = spell.unknown(words)
    return " ".join([spell.correction(w) if w in misspelled else w for w in words])

def clean_text_aggressive(text):
    """
    Cleaning 1 (Notebook Methodology): 
    - Hyphen preservation (customer-service -> customer_service)
    - Stopword removal (Generic + Domain)
    - SpaCy Lemmatization
    """
    if not isinstance(text, str): return ""
    
    # 1. Lowercase & URL/HTML removal
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+|<[^>]+>", " ", text)
    
    # 2. Preserve hyphenated compounds before stripping punctuation
    text = re.sub(r"(\w)-(\w)", r"\1_\2", text)
    text = re.sub(r"[^a-z0-9_ ]", " ", text)
    
    # 3. Domain Stopwords
    stop_words = set(stopwords.words('english'))
    domain_stops = {
        "insurance", "insurer", "assurance", "assureur", "contract", "contrat", 
        "policy", "guarantee", "cover", "coverage", "subscriber", "would", 
        "also", "get", "told", "said", "take", "make", "give", "review", "customer", "company"
    }
    stop_words.update(domain_stops)
    
    # 4. SpaCy Lemmatization (Much better than NLTK WordNet)
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if token.text not in stop_words and len(token.text) > 1]
    
    return " ".join(lemmas)

def clean_text_preservative(text):
    """Cleaning 2: Preservative for Transformers. Keep stopwords and punctuation."""
    if not isinstance(text, str): return ""
    return text.strip()

def run_full_pipeline(df):
    """Executes the optimized 'Double Clean' pipeline."""
    logging.info("Starting Optimized Data Engineering Pipeline...")
    
    # 1. Recover missing English text
    df = recover_placeholders(df)
    
    # 2. Spelling Correction (with Domain Guard)
    domain_guard = ["smabtp", "olivier", "direct", "assurance", "allianz", "axa", "generali"]
    logging.info("Applying Spelling Correction...")
    df['avis_corrected'] = df['avis_en'].apply(lambda x: correct_spelling(x, domain_guard))
    
    # 3. Double Clean Strategy
    logging.info("Generating 'Cleaning 1' (Aggressive/SpaCy)...")
    df['avis_cleaning_1'] = df['avis_corrected'].apply(clean_text_aggressive)
    
    logging.info("Generating 'Cleaning_2' (Preservative)...")
    df['avis_cleaning_2'] = df['avis_corrected'].str.strip()
    
    # Keep metadata for exported schema
    final_cols = [
        'note', 'assureur', 'produit', 'type', 'date_publication', 'date_exp',
        'avis_en', 'avis_corrected', 'avis_cleaning_1', 'avis_cleaning_2'
    ]
    
    return df[final_cols]

if __name__ == "__main__":
    DATA_DIR = "/home/leopa/projects/nlp2/data"
    raw_df = load_all_data(DATA_DIR)
    # Process a sample or full based on environment
    clean_df = run_full_pipeline(raw_df.head(100)) 
    print(clean_df.head())

