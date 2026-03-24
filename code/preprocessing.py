import pandas as pd
import numpy as np
import re
import os
import glob
import logging
import spacy
import language_tool_python

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    nlp = spacy.load("fr_core_news_sm", disable=["parser", "ner"])
    stop_words_fr = set(nlp.Defaults.stop_words)
    stop_words_fr.update(['assurance', 'assureur', 'contrat', 'mutuelle', 'client', 'service', 'dossier', 'année', 'mois'])
except OSError:
    logging.error("SpaCy model 'fr_core_news_sm' not found. Run 'python3 -m spacy download fr_core_news_sm'")
    nlp = None
    stop_words_fr = set()

def load_all_data(data_dir):
    """Loads all 35 .xlsx files and cleans column names."""
    files = glob.glob(os.path.join(data_dir, "*.xlsx"))
    dfs = []
    for f in files:
        dfs.append(pd.read_excel(f))
    if not dfs:
        return pd.DataFrame()
    full_df = pd.concat(dfs, ignore_index=True)
    full_df.columns = full_df.columns.str.strip().str.lower().str.replace(r"[\s/]+", "_", regex=True)
    logging.info(f"Loaded {len(full_df)} rows. Columns normalized.")
    return full_df

def correct_with_languagetool(text, tool):
    """Uses language_tool_python to correct spelling while ignoring protected words (insurers)."""
    if not isinstance(text, str) or len(text) < 3:
        return text
    
    protected_words = {"smabtp", "olivier", "direct", "allianz", "axa", "generali", "april", "macif", "maaf", "gmf", "santiane"}
    
    matches = tool.check(text)
    filtered_matches = []
    for match in matches:
        error_context = text[match.offset : match.offset + match.errorLength].lower()
        if error_context.strip() in protected_words:
            continue
        filtered_matches.append(match)
        
    return language_tool_python.utils.correct(text, filtered_matches)

def clean_text(text, lemmatize=False):
    """Cleans French text (lowercase, regex out HTML/URLs, lemmatize, remove stopwords)."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+|<[^>]+>", " ", text)
    text = re.sub(r"[^a-zàâäéèêëîïôûùûüçœæ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words_fr and len(t) > 2]

    if lemmatize and nlp is not None:
        doc = nlp(" ".join(tokens))
        tokens = [token.lemma_ for token in doc if not token.is_space]
        
    return " ".join(tokens)

def run_full_pipeline(df):
    """Executes the preprocessing pipeline from the notebook."""
    logging.info("Starting Preprocessing Pipeline...")
    
    # Text spelling correction (French)
    tool = language_tool_python.LanguageTool('fr')
    
    logging.info("Correcting spelling...")
    df['avis_corrected'] = df['avis'].apply(lambda x: correct_with_languagetool(x, tool))
    
    logging.info("Cleaning and lemmatizing text...")
    df['avis_corrected_clean'] = df['avis_corrected'].apply(clean_text)
    
    final_cols = ['note', 'assureur', 'produit', 'type', 'date_publication', 'avis', 'avis_corrected', 'avis_corrected_clean']
    # keep extra columns if they exist
    for col in df.columns:
        if col not in final_cols:
            final_cols.append(col)
            
    return df[final_cols]
