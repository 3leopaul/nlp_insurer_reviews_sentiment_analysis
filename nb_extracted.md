--- CELL 0 (markdown) ---
# NLP Project 2, Supervised Learning on Insurance Reviews

**Deadline:** 31 mars 2026  
**Dataset:** French insurance reviews (multi-file Excel dataset in `data/`)  
**LLM used:** `Qwen2.5-1.5B-Instruct`

---

## Table of Contents
1. [Environment Setup & Imports](#1)
2. [Data Loading & Initial Exploration](#2)
3. [Section 1, Data Cleaning](#3)
   - 3.1 Basic Cleaning
   - 3.2 Frequent Words & N-grams
   - 3.3 Spelling Correction
4. [Section 2, Summary, Translation & Generation](#4)
5. [Section 3, Topic Modeling](#5)
6. [Section 4, Embeddings & Similarity](#6)
   - 6.1 Word2Vec Training
   - 6.2 GloVe
   - 6.3 Matplotlib Visualization
   - 6.4 TensorBoard Visualization
   - 6.5 Distance Metrics
   - 6.6 Semantic Search (Bonus)
7. [Section 5, Supervised Learning](#7)
   - 7.1 TF-IDF + Classical ML
   - 7.2 Basic Embedding Layer (Keras)
   - 7.3 Pre-trained Embeddings
   - 7.4 BERT / HuggingFace Transformers
   - 7.5 LLM (Qwen2.5-1.5B-Instruct)
   - 7.6 Model Comparison
8. [Section 6, Results Interpretation](#8)
   - 8.1 Error Analysis
   - 8.2 Sentiment Detection
   - 8.3 Classical Models with Themes
   - 8.4 Deep Learning Models for Words
9. [Section 7, Streamlit Applications](#9)
   - 9.1 Prediction App
   - 9.2 Summary App
   - 9.3 Explanation App (SHAP/LIME)
   - 9.4 Information Retrieval App
   - 9.5 RAG App
   - 9.6 QA App

---

--- CELL 1 (markdown) ---
<a id='1'></a>
## 1. Environment Setup & Imports

We install all required libraries and configure the environment. The project relies on:
- `pandas`, `numpy`, `matplotlib`, `seaborn` for data manipulation and visualization
- `nltk`, `spacy`, `textblob`, `langdetect` for NLP preprocessing
- `gensim` for Word2Vec and GloVe loading
- `scikit-learn` for classical ML models and evaluation
- `tensorflow`/`keras` for deep learning models
- `transformers` (HuggingFace) for BERT and LLM (Qwen2.5)
- `sentence-transformers` for USE-equivalent embeddings
- `pyLDAvis`, `gensim` for topic modeling
- `shap`, `lime` for explainability
- `faiss-cpu` for semantic search
- `pyspellchecker` for spelling correction

--- CELL 2 (code) ---
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

--- CELL 3 (code) ---
# ─── Core imports ─────────────────────────────────────────────────────────────
import os, re, json, glob, warnings, logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from collections import Counter
from pathlib import Path

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.ERROR)
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({'figure.dpi': 120, 'figure.figsize': (12, 5)})

# ─── NLTK resources ───────────────────────────────────────────────────────────
import nltk
for resource in ['stopwords', 'punkt', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger']:
    nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

# ─── NLP ──────────────────────────────────────────────────────────────────────
import spacy
nlp_fr = spacy.load('fr_core_news_sm')
nlp_en = spacy.load('en_core_web_sm')

from textblob import TextBlob
from langdetect import detect
from spellchecker import SpellChecker
import language_tool_python
from tqdm import tqdm

print('✅ Core imports OK')

--- CELL 4 (code) ---
# ─── ML / DL imports ──────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder

print('✅ Sklearn imports OK')

--- CELL 5 (code) ---
# ─── Gensim / Topic Modeling ──────────────────────────────────────────────────
import gensim
from gensim.models import Word2Vec, KeyedVectors
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from wordcloud import WordCloud

print('✅ Gensim imports OK')

--- CELL 6 (code) ---
# ─── TensorFlow / Keras ────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Embedding, LSTM, GRU, Conv1D, GlobalMaxPooling1D,
    Dense, Dropout, Bidirectional, Input, Flatten
)
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.utils import to_categorical

print(f'✅ TensorFlow {tf.__version__} OK')

--- CELL 7 (code) ---
# ─── HuggingFace Transformers ────────────────────────────────────────────────
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForCausalLM, pipeline, TextClassificationPipeline
)
from sentence_transformers import SentenceTransformer
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'✅ Transformers OK  |  Device: {DEVICE}')

--- CELL 8 (code) ---
# ─── Explainability & Semantic Search ────────────────────────────────────────
import shap
import lime
from lime.lime_text import LimeTextExplainer
import faiss

print('✅ SHAP, LIME, FAISS OK')

--- CELL 9 (code) ---
# ─── GPU Configuration ────────────────────────────────────────────────────────
# Allow GPU memory to grow incrementally rather than allocating all at once
# This prevents OOM crashes when multiple models share the GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f'✅ GPU memory growth enabled for {len(gpus)} GPU(s)')
        print(f'   Device: {gpus[0].name}')
    except RuntimeError as e:
        # Must be set before GPUs are initialised
        print(f'⚠️ GPU config error: {e}')
else:
    print('No GPU detected, running on CPU')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress INFO and WARNING logs
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # suppress HuggingFace warning

--- CELL 10 (markdown) ---
<a id='2'></a>
## 2. Data Loading & Initial Exploration

The dataset is spread across multiple Excel files in the `data/` directory. Each file contains insurance reviews with the following columns:

| Column | Description |
|---|---|
| `note` | Star rating (1-5) |
| `auteur` | Reviewer username |
| `avis` | Original French review text |
| `assureur` | Insurance company name |
| `produit` | Insurance product type |
| `type` | Train/test split indicator |
| `date_publication` | Publication date |
| `date_exp` | Experience date |
| `avis_en` | English translation of review |
| `avis_cor` | Corrected French review |
| `avis_cor_en` | Corrected English review |

--- CELL 11 (code) ---
# ─── Load all Excel files from data/ directory ────────────────────────────────
DATA_DIR = Path('data')
excel_files = sorted(DATA_DIR.glob('*.xlsx')) + sorted(DATA_DIR.glob('*.xls'))

print(f'Found {len(excel_files)} Excel file(s):')
for f in excel_files:
    print(f'  • {f.name}')

dfs = []
for f in excel_files:
    try:
        tmp = pd.read_excel(f, engine='openpyxl')
        tmp['source_file'] = f.name
        dfs.append(tmp)
        print(f'  ✓ {f.name} → {len(tmp)} rows, {tmp.shape[1]} cols')
    except Exception as e:
        print(f'  ✗ {f.name} → ERROR: {e}')

df_raw = pd.concat(dfs, ignore_index=True)
print(f'\nTotal rows after merging: {len(df_raw):,}')

--- CELL 12 (code) ---
# ─── Preview and schema inspection ───────────────────────────────────────────
print('=== Head ===')
display(df_raw.head(3))

print('\n=== Data types ===')
print(df_raw.dtypes)

print('\n=== Missing values ===')
missing = df_raw.isnull().sum()
print(missing[missing > 0])

print('\n=== Shape ==', df_raw.shape)

--- CELL 13 (markdown) ---
#### Shape & Scale

We have 34,435 reviews across 12 columns.

#### Data Types

The most glaring issue is that 'note' (the star rating) is float64 and date_publication/date_exp are object rather than datetime. This means the dates were either inconsistently formatted across source files or contain non-parseable values. We'll want to coerce those aggressively during cleaning and inspect why note insn't simply in int64.

#### Missing Values, the Critical Finding

This is the most important signal from the EDA output. Three distinct patterns:

note has 10,331 missing values, roughly 30% of the entire dataset. This is a severe problem for any supervised star-rating prediction task.

--- CELL 14 (code) ---
df_raw['type'].unique()

--- CELL 15 (code) ---
note_stats = df_raw.groupby('type')['note'].agg(
    Null_Values=lambda x: x.isnull().sum(),
    Non_Null_Values=lambda x: x.notnull().sum()
)
print(note_stats)

--- CELL 16 (markdown) ---
After inspecting the 'type' column, we find that the missing 'note' values simply correspond to the "test" type which means that the dataset has previously been split into train and test sets and the note missing will only need to be filled.

--- CELL 17 (code) ---
df_raw['note'].unique()

--- CELL 18 (markdown) ---
Testing if the note values are indeed comprised between 1 and 5 returns positive results, there are no star reviews under 1 or above 5 except for the null star review values which correspond to the test set.

--- CELL 19 (code) ---
df_test = df_raw.copy()
# 1. Prepare Data
# Create a boolean column for missing notes
df_test['is_missing'] = df_test['note'].isna()
# Calculate count and proportion per insurer
stats = df_test.groupby('assureur')['is_missing'].agg(
    Missing_Count='sum',
    Missing_Proportion='mean'
).reset_index()
# Sort by count for better visualization
stats = stats.sort_values('Missing_Count', ascending=False)
# 2. Plotting - Stacking vertically
fig, axes = plt.subplots(2, 1, figsize=(14, 14))
# --- Plot 1: Count of Missing Notes ---
sns.barplot(data=stats, x='assureur', y='Missing_Count', ax=axes[0], palette='Blues_r')
axes[0].set_title('Total Count of Missing Notes per Insurer (Test Split)', fontsize=15, fontweight='bold', pad=20)
axes[0].set_ylabel('Number of Missing Reviews')
axes[0].set_xlabel('Insurer')
axes[0].tick_params(axis='x', rotation=45)
# --- Plot 2: Proportion of Missing Notes ---
sns.barplot(data=stats, x='assureur', y='Missing_Proportion', ax=axes[1], palette='Reds_r')
axes[1].set_title('Proportion of Missing Notes per Insurer (Relative to Total)', fontsize=15, fontweight='bold', pad=20)
axes[1].set_ylabel('Proportion (1.0 = 100%)')
axes[1].set_xlabel('Insurer')
axes[1].tick_params(axis='x', rotation=45)
# Add extra space between plots
plt.subplots_adjust(hspace=0.6)
plt.tight_layout()
plt.show()

--- CELL 20 (code) ---
df_test = df_raw.copy()
# 1. Prepare Data
# Create a boolean column for missing notes
df_test['is_missing'] = df_test['note'].isna()
# Calculate count and proportion per insurer
stats = df_test.groupby('produit')['is_missing'].agg(
    Missing_Count='sum',
    Missing_Proportion='mean'
).reset_index()
# Sort by count for better visualization
stats = stats.sort_values('Missing_Count', ascending=False)
# 2. Plotting - Stacking vertically
fig, axes = plt.subplots(2, 1, figsize=(14, 14))
# --- Plot 1: Count of Missing Notes ---
sns.barplot(data=stats, x='produit', y='Missing_Count', ax=axes[0], palette='Blues_r')
axes[0].set_title('Total Count of Missing Notes per Product (Test Split)', fontsize=15, fontweight='bold', pad=20)
axes[0].set_ylabel('Number of Missing Reviews')
axes[0].set_xlabel('Product')
axes[0].tick_params(axis='x', rotation=45)
# --- Plot 2: Proportion of Missing Notes ---
sns.barplot(data=stats, x='produit', y='Missing_Proportion', ax=axes[1], palette='Reds_r')
axes[1].set_title('Proportion of Missing Notes per Product (Relative to Total)', fontsize=15, fontweight='bold', pad=20)
axes[1].set_ylabel('Proportion (1.0 = 100%)')
axes[1].set_xlabel('Product')
axes[1].tick_params(axis='x', rotation=45)
# Add extra space between plots
plt.subplots_adjust(hspace=0.6)
plt.tight_layout()
plt.show()

--- CELL 21 (markdown) ---
After anylysing the distribution of missing values accross products and insurers, We notice that the vast majority of mainstream insurers and products sit in a tight band of ~28–33% missing notes. This indicates that the test/train split was well done and isn't skewed towards one category. We will however need to be careful with the insurer Hiscox and the product flotte-automobile as they have a much higher proportion of missing notes.

--- CELL 22 (code) ---
flotte_count = len(df_raw[df_raw['produit'] == 'flotte-automobile'])
print(f"Number of reviews for 'flotte-automobile': {flotte_count}")

--- CELL 23 (code) ---
hiscox_count = len(df_raw[df_raw['assureur'] == 'Hiscox'])
print(f"Number of reviews for 'Hiscox': {hiscox_count}")

--- CELL 24 (markdown) ---
Further annalysis of the two proportional anomalies in missing note values reveals that they are due to very low review count in these categories. It is therefore not a concern for the model training process. We will also exclude these reviews from any per assurer or per product analysis and input a warning on the streamlit applications for these categories all the while showing the existing reviews.

--- CELL 25 (markdown) ---
'auteur' has just 1 missing value, essentially negligible, likely a single anonymised or bot submission. No action needed.

--- CELL 26 (code) ---
print(df_raw[df_raw['auteur'].isna()])

--- CELL 27 (markdown) ---
The only missing auteur is from a deleted review.

avis_en has 2 missing values, near-complete translation coverage, which is excellent, we will therefore only need to translate 2 reviews.

avis_cor and avis_cor_en have ~34,000 missing values, practically the entire dataset. This tells us the corrected columns were either never populated at source, or only filled for a tiny subset. These columns should be treated as nearly empty and we should generate them ourselves.

--- CELL 28 (code) ---
# ─── Distribution of star ratings ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Star distribution
df_raw['note'].value_counts().sort_index().plot.bar(ax=axes[0], color='steelblue', edgecolor='white')
axes[0].set_title('Distribution of Star Ratings')
axes[0].set_xlabel('Stars')
axes[0].set_ylabel('Count')

# Top insurers
top_insurers = df_raw['assureur'].value_counts().head(10)
top_insurers.plot.barh(ax=axes[1], color='coral', edgecolor='white')
axes[1].set_title('Top 10 Insurers by Review Count')
axes[1].invert_yaxis()

# Product distribution
df_raw['produit'].value_counts().head(8).plot.pie(
    ax=axes[2], autopct='%1.1f%%', startangle=140
)
axes[2].set_title('Insurance Product Distribution')
axes[2].set_ylabel('')

plt.tight_layout()
plt.savefig('outputs/eda_overview.png', bbox_inches='tight')
plt.show()
print('EDA overview saved.')

--- CELL 29 (markdown) ---
### Star Rating Distribution

The distribution is notably bimodal and asymmetric. The dominant pattern is a heavy spike at 1-star (~7,200 reviews) followed by a secondary peak at 4 and 5 stars (~4,900 each), with 2 and 3 stars forming a valley in between. This is a classic J-shaped distribution extremely common in online review platforms, where customers are most motivated to write a review either when they are very unhappy or genuinely delighted, while middling experiences rarely inspire anyone to take the time.

The practical consequence is significant for modelling. The negative class is roughly 1.5× larger than either positive bucket, meaning the dataset is moderately imbalanced. A naive classifier could exploit the 1-star dominance and still achieve decent accuracy. We should either apply class weighting, use stratified sampling, or report F1-macro rather than raw accuracy to get a fair picture of model performance across all sentiment classes. The neutral class (3-star) being the smallest bucket will be the hardest to predict reliably.

### Top 10 Insurers by Review Count (Middle)

The insurer distribution is heavily concentrated at the top. Direct Assurance (~8,200) and L'Olivier Assurance (~6,000) together likely account for a substantial fraction of the entire corpus, with a sharp drop-off to APRIL Moto, GMF and the rest sitting well below 2,000 reviews each. This is a long-tail distribution, the bottom 40+ insurers in the full dataset will each have very few reviews. The dominance of Direct Assurance and L'Olivier is not surprising, both are major French insurers with high digital engagement, which naturally drives more platform reviews. However, it means that per-insurer analyses and model evaluations will be statistically robust only for these top players. Any aggregate metric reported will be heavily influenced by how well the model performs on these two insurers specifically.

### Insurance Product Distribution (Right)

Auto insurance accounts for 58.6% of all reviews, an overwhelming majority. Santé (health) follows at 14.6% and moto at 8.8%, with habitation, prévoyance, crédit, vie and animaux making up small single-digit slices. This extreme product imbalance has several implications.

Any model trained on this corpus without product-level stratification will essentially be an auto insurance sentiment model with some generalisation to other products. If our goal is to perform well across all product types, we should consider either training separate models per product for the top 3-4 categories, or including `produit` as a feature. Cross-product evaluation metrics will be dominated by auto performance and will mask potentially poor generalisation to health or home insurance language, which uses quite different vocabulary and concerns.

--- CELL 30 (markdown) ---
<a id='3'></a>
## 3. Section 1, Data Cleaning (2 points)

Data cleaning is foundational. A poorly-cleaned corpus will cascade errors into every downstream task. We perform the following steps:

1. **Structural cleaning** – drop duplicates, fix data types, normalise column names
2. **Text normalisation** – lowercase, strip HTML/URLs/special characters, expand contractions
3. **Tokenisation & stopword removal**
4. **Lemmatisation** with spaCy
5. **Frequent words & N-gram analysis**
6. **Spelling correction** with PySpellChecker

--- CELL 31 (markdown) ---
### 3.1 Structural Cleaning

--- CELL 32 (code) ---
# ─── Make a working copy ─────────────────────────────────────────────────────
df = df_raw.copy()

# Normalise column names
df.columns = df.columns.str.strip().str.lower()

# Parse dates
for col in ['date_publication', 'date_exp']:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')

# Cast rating to numeric
df['note'] = pd.to_numeric(df['note'], errors='coerce')

# Remove duplicate reviews (same author + same text)
n_before = len(df)
df.drop_duplicates(subset=['auteur', 'avis'], keep='first', inplace=True)
n_after = len(df)
print(f'Removed {n_before - n_after:,} duplicate rows. Remaining: {n_after:,}')

# Drop rows with empty review text
df.dropna(subset=['avis'], inplace=True)
df = df[df['avis'].str.strip().ne('')]
df.reset_index(drop=True, inplace=True)
print(f'After dropping empty reviews: {len(df):,}')

# Ensure note is within valid range (except for test set)
df = df[(df['note'].between(1, 5)) | (df['type'] == 'test')]
df.reset_index(drop=True, inplace=True)
print(f'After filtering (Valid ratings 1-5 + all Test rows): {len(df):,}')

--- CELL 33 (markdown) ---
### 3.2 Text Normalisation Pipeline

--- CELL 34 (code) ---
# ─── Text normalisation helpers ───────────────────────────────────────────────
FR_STOP = set(stopwords.words('french'))
EN_STOP = set(stopwords.words('english'))
ALL_STOP = FR_STOP | EN_STOP

def clean_text(text: str, remove_stops: bool = True, lemmatize: bool = False, keep_apostrophes: bool = False) -> str:
    """Full text cleaning pipeline for French/English insurance reviews."""
    if not isinstance(text, str):
        return ''
    # Lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    # Remove emails
    text = re.sub(r'\S+@\S+', ' ', text)

    # Remove non-alphabetic characters (keep spaces)
    if keep_apostrophes:
        text = text.replace('\u2019', "'")  # normalise curly apostrophe first
        text = re.sub(r"[^a-zàâäéèêëîïôùûüçœæ'\s]", ' ', text)
    else:
        text = re.sub(r"[^a-zàâäéèêëîïôùûüçœæ\s]", ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Tokenise
    tokens = text.split()
    # Remove stopwords
    if remove_stops:
        tokens = [t for t in tokens if t not in ALL_STOP and len(t) > 2]
    # Lemmatise with spaCy (optional, slower)
    if lemmatize:
        doc = nlp_fr(' '.join(tokens))
        tokens = [token.lemma_ for token in doc if not token.is_space]
    return ' '.join(tokens)


def clean_text_keep_stops(text: str) -> str:
    """Light cleaning (keep stopwords, useful for sentiment models)."""
    return clean_text(text, remove_stops=False, lemmatize=False, keep_apostrophes=True)


# Apply to both French and English columns
print('Cleaning avis (French)...')
df['avis_clean'] = df['avis'].apply(clean_text)

print('Cleaning avis_en (English translations)...')
avis_en_col = 'avis_en' if 'avis_en' in df.columns else 'avis'
df['avis_en_clean'] = df[avis_en_col].fillna(df['avis']).apply(clean_text)

print('Cleaning with stops preserved...')
df['avis_light'] = df['avis'].apply(clean_text_keep_stops)

print('Done!')
display(df[['avis', 'avis_clean', 'avis_en_clean', 'avis_light']].head(5))



--- CELL 35 (markdown) ---
This is an important step especially for TF IDF in which we implement multiple transformations to the original text for multiple reasons:

- Lowercase
Ensures "Assurance", "ASSURANCE" and "assurance" are treated as the same token. Without this, the vocabulary size explodes with capitalisation variants that carry no additional meaning. Particularly important for TF-IDF where "Satisfait" and "satisfait" would otherwise be counted separately.

- Remove HTML tags
Reviews scraped from the web may contain residual HTML markup like "&lt;br&gt;", "&lt;p&gt;", "&lt;strong&gt;" etc. These carry no semantic content and would appear as garbage tokens in the vocabulary.

- Remove URLs
Some reviewers paste links to supporting documents or insurer websites. URLs fragment into meaningless tokens like "http", "www", "com" that would inflate the vocabulary with noise.

- Remove emails
Same reasoning as URLs, email addresses appearing in reviews (e.g. contact addresses for insurers) would produce uninformative tokens.

- Normalise curly apostrophes and character filtering
The curly apostrophe normalisation ensures web-scraped text using typographic quotes is handled consistently. The character filter then removes everything that isn't a French letter, apostrophe, or whitespace, this strips numbers, punctuation, special characters and symbols. Numbers like "270 euros" or "3 mois" lose their numeric component but the surrounding words are preserved. Punctuation like commas, exclamation marks and periods are removed since they carry limited value for bag-of-words models and would otherwise create tokens like "satisfait." and "satisfait" as separate vocabulary entries.

- Collapse whitespace
The previous substitutions replace removed characters with spaces, which can produce multiple consecutive spaces. This collapses them back into single spaces and strips leading/trailing whitespace.

- Tokenise
Splits the cleaned string into a list of individual word tokens for the subsequent filtering steps.

- Remove stopwords and short tokens
Two filters applied simultaneously. Stopwords, from a combined French and English list, are removed because they appear in virtually every document and provide no discriminative signal for classification or topic modelling. The len(t) > 2 filter removes single characters and two-character tokens like "j'", "ai", "du", "la" that survived stopword removal, these are either contraction fragments or very short function words that add noise. Note that this ALL_STOP list combines both French and English stopwords because avis_en_clean uses the same function on translated text.

- Lemmatisation (optional, disabled by default)
When enabled, reduces each word to its dictionary base form using spaCy, "satisfaits", "satisfaite", "satisfaisante" would all become "satisfait". This reduces vocabulary size and consolidates related forms. It is disabled by default because it is significantly slower and the benefit for TF-IDF models on this corpus is marginal given that French morphological variants of the same root tend to co-occur frequently enough to be treated similarly by the model anyway.
When enabled, reduces each word to its dictionary base form using spaCy, "satisfaits", "satisfaite", "satisfaisante" would all become "satisfait". This reduces vocabulary size and consolidates related forms. It is disabled by default because it is significantly slower and the benefit for TF-IDF models on this corpus is marginal given that French morphological variants of the same root tend to co-occur frequently enough to be treated similarly by the model anyway.

### What avis_clean is good for and what it isn't
The result is a bag of meaningful content words with no punctuation, no stopwords, no numbers, suitable for TF-IDF vectorisation, Word2Vec training, and topic modelling. It is not suitable for sentiment models that need grammatical context, negations like "n'est pas satisfait" become just "satisfait" after stopword removal, flipping the sentiment entirely. That is why avis_light exists as a separate column.


### PS: for now we apply normalisation to uncorrected reviews, but we will later re-apply the same normalisation to corrected reviews. This is because the correction model uses punctuation and spacing to make corrections, so we need to keep the original text for the correction model.

--- CELL 36 (markdown) ---
### 3.3 Frequent Words & N-grams

Analysing the most common terms and bigrams/trigrams gives us insight into recurring themes in the reviews, pricing, customer service, claims, etc.

--- CELL 37 (code) ---
# ─── Top unigrams (Original and English) ──────────────────────────────────────
# List the columns you want to visualize
columns_to_analyze = ['avis_clean', 'avis_en_clean']
for col in columns_to_analyze:
    # Skip if column doesn't exist in df
    if col not in df.columns:
        print(f"Column {col} not found in DataFrame.")
        continue
        
    all_tokens = ' '.join(df[col].dropna()).split()
    word_freq = Counter(all_tokens)
    top_words = pd.DataFrame(word_freq.most_common(30), columns=['word', 'freq'])
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    # Bar chart
    # We use a different color for the English version to distinguish them
    bar_color = 'steelblue' if 'en' not in col else 'seagreen'
    axes[0].barh(top_words['word'][::-1], top_words['freq'][::-1], color=bar_color)
    axes[0].set_title(f'Top 30 Most Frequent Words ({col})')
    axes[0].set_xlabel('Frequency')
    # Word cloud
    # We use a different colormap for the English version
    wc_cmap = 'Blues' if 'en' not in col else 'Greens'
    wc = WordCloud(
        width=600, height=400, background_color='white',
        colormap=wc_cmap, max_words=150
    ).generate_from_frequencies(word_freq)
    
    axes[1].imshow(wc, interpolation='bilinear')
    axes[1].axis('off')
    axes[1].set_title(f'Word Cloud, Insurance Reviews ({col})')
    plt.tight_layout()
    
    # Save with a unique filename for each column
    plt.savefig(f'outputs/wordcloud_unigrams_{col}.png', bbox_inches='tight')
    plt.show()

--- CELL 38 (markdown) ---
For the Frequent words we look at the the french and english word clouds and compare them.

### Overall Similarity Confirms Translation Fidelity
The two distributions are remarkably parallel. "assurance/insurance", "service/service", "prix/price", "contrat/contract" map directly to each other and maintain roughly the same relative frequencies. This tells us the translation is semantically faithful at the corpus level, the machine translation preserved topic distribution rather than introducing systematic drift. That's reassuring for our future GloVe experiments.

#### Stopword Leakage is Worse in French
The French chart has a significant stopword leakage problem that the English chart does not. "Très", "plus", "bien", "tout", "depuis", "fait", "chez", "faire", "sans", "avoir", "car", "donc", "après", "rien" all appear in the French top 30 and carry almost no discriminative power as unigrams. These are French quasi-stopwords that our NLTK stopword list missed, When looking into it, NLTK's French is indeed considered incomplete compared to its English equivalent. "Très" (very), "plus" (more/no longer), "bien" (well/good), "tout" (all/everything), "depuis" (since/for) are essentially adverbs and discourse markers that survived because they aren't in the standard list.

We should build a custom French stopword extension and add at minimum: très, plus, bien, tout, depuis, fait, chez, faire, sans, avoir, car, donc, après, rien, cette, toujours, ans. This will meaningfully clean our Word2Vec training corpus and our TF-IDF feature space.

"Sans" and "Rien" are Valuable Exceptions
Worth noting that "sans" (without) and "rien" (nothing) do carry sentiment signal despite being quasi-stopwords, they mirror the English "without" and "nothing" identified earlier as negative sentiment proxies. If we strip them entirely we lose that signal. The better approach is to keep them for sentiment modelling but remove them from our topic modelling and Word2Vec training where they add noise without semantic content.

### French Reveals Temporal Loyalty vs. Complaint Pattern More Clearly
"Depuis" and "mois" appearing prominently in French, alongside "ans" (years), confirms the temporal anchoring pattern identified earlier. Interestingly "depuis" is more nuanced than its English translation "since", in French reviews it typically introduces either long loyalty ("depuis 10 ans client") or protracted grievance ("depuis 3 mois sans réponse"). This ambiguity is worth exploiting as a bigram feature: "depuis ans" will skew positive while "depuis mois" will skew negative in most contexts.

### "Sinistre" Appearing in French but Not English Top 30
"Sinistre" (insurance claim/incident) appears in the French top 30 but its English equivalent "incident" doesn't make the English top 30. This is a translation artefact, "sinistre" gets translated inconsistently as "incident", "claim", "disaster", or "damage" depending on context, fragmenting its frequency across multiple English tokens. This is a concrete example of why training our primary embedding models on French is the right call, the semantic unit "sinistre" is coherent in French but dissolved in translation.

--- CELL 39 (code) ---
# ─── N-grams (Bigrams & Trigrams) - French and English ────────────────────────

def get_ngram_freq(corpus_tokens: list, n: int = 2) -> Counter:
    """Compute n-gram frequency from a list of tokens."""
    from nltk import ngrams # Ensure ngrams is imported
    ng = [gram for token_list in corpus_tokens for gram in ngrams(token_list, n)]
    return Counter(ng)

def plot_ngrams(freq: Counter, n: int, col_name: str, top_k: int = 20):
    label = 'Bigrams' if n == 2 else 'Trigrams'
    suffix = "French" if col_name == 'avis_clean' else "English"
    
    top = pd.DataFrame(
        [(' '.join(g), c) for g, c in freq.most_common(top_k)],
        columns=['ngram', 'freq']
    )
    
    fig, ax = plt.subplots(figsize=(12, 5))
    # 'coral' for French, 'salmon' for English
    color = 'coral' if col_name == 'avis_clean' else 'salmon'
    
    ax.barh(top['ngram'][::-1], top['freq'][::-1], color=color)
    ax.set_title(f'Top {top_k} {label} ({suffix})')
    ax.set_xlabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f'outputs/ngrams_{label.lower()}_{col_name}.png', bbox_inches='tight')
    plt.show()
    return top

# Loop through columns: avis_clean and avis_en_clean
for col in ['avis_clean', 'avis_en_clean']:
    print(f"\n{'='*40}\n ANALYZING: {col}\n{'='*40}")
    
    # Pre-tokenize
    tokenised = [sent.split() for sent in df[col].dropna() if sent.strip()]

    # Compute frequencies
    bigram_freq  = get_ngram_freq(tokenised, 2)
    trigram_freq = get_ngram_freq(tokenised, 3)

    # Plot and display
    top_bigrams  = plot_ngrams(bigram_freq, 2, col)
    top_trigrams = plot_ngrams(trigram_freq, 3, col)
    
    print(f'Top bigrams ({col}):'); display(top_bigrams.head(10))
    print(f'\nTop trigrams ({col}):'); display(top_trigrams.head(10))


--- CELL 40 (markdown) ---
### French Bigrams, Strong Sentiment Polarity Signal
The top French bigrams split cleanly into three functional categories:

- The positive sentiment cluster is immediately visible: "satisfait service", "très satisfait", "très bien", "très bon", "qualité prix", "bon rapport" all appear in the top 20. These are reliable positive class markers and will be powerful TF-IDF features. "Rapport qualité prix" (value for money) deserves special attention, it's the single most frequent trigram at 545 occurrences, meaning French reviewers explicitly invoke the value proposition more than any other concept. This is a strong prior that pricing satisfaction is the primary driver of positive reviews.

- The negative sentiment cluster is more subtle at the bigram level but becomes explicit in trigrams: "sinistre non responsable" and "accident non responsable" both appear, referring to the French insurance concept of a "non-responsible claim" where the insured party was not at fault. These appear frequently because customers who were involved in accidents they didn't cause often write detailed complaints about how the claim was handled. These are almost exclusively negative or frustrated reviews.

- The temporal loyalty cluster, "depuis ans", "plus ans", "depuis plus", "aujourd hui", "depuis plusieurs années" confirms the pattern identified earlier. "Depuis plus ans" as the second most frequent trigram (506 occurrences) is particularly telling: customers frequently frame their dissatisfaction or satisfaction relative to long tenure, implying betrayal of loyalty or consistency rather than a first-time bad or good experience.

### French Trigrams, Most Analytically Valuable
"Rapport qualité prix" at 545 and "depuis plus ans" at 506 being the two dominant trigrams reveals that the corpus is actually top-heavy with positive reviews at the n-gram level. Both trigrams are predominantly positive signals, the first expressing satisfaction with value for money, the second reflecting long-term loyal customers. This is seemingly at odds with the 1-star spike seen in the rating distribution, but is reconciled by the fact that dissatisfied customers tend to write longer, more elaborate reviews that generate a wider variety of n-grams rather than converging on a single dominant phrase. Negative sentiment is therefore more lexically diverse and diffuse, while positive sentiment is more formulaic and concentrated, a pattern with direct implications for our TF-IDF models, where positive class features will be tighter and more reliable than negative ones.

" Chez direct assurance" (335) and "recommande direct assurance" / "chez olivier assurance" confirm that Direct Assurance and L'Olivier dominate not just in volume but in named mentions within review text, reviewers explicitly name these insurers in their text body, which could cause data leakage if we're not careful: a model trained with insurer names as features will learn to associate "direct assurance" with a particular sentiment distribution rather than learning transferable language patterns.

### English Bigrams, Translation Artefacts Are Visible

"Olive tree" appearing as a top-20 English bigram is the most glaring finding here and a direct translation error. "L'Olivier Assurance", named after the olive tree, was literally translated as "olive tree" instead of being preserved as a proper noun. This means every mention of L'Olivier in the English corpus is fragmented across "olivier", "olive", "tree", "olive tree" rather than being a single coherent token. This is a concrete illustration of why English should not be our primary modelling language and why proper noun handling matters in our preprocessing.

"Green card" appearing is another interesting artefact, in French auto insurance "carte verte" (green card) is the proof of insurance document, so this is a legitimate domain term rather than a political reference. It confirms the auto insurance dominance of the corpus vocabulary.
"Third party" maps to "tiers" in French auto insurance (third-party liability), again confirming auto-specific terminology dominates.

### English Trigrams, Negative Signal Fragmentation

"Non responsible accident" and "non responsible disaster" both appearing confirms the sinistre non responsable pattern survives translation, though "disaster" as a translation of "sinistre" is slightly dramatic and illustrates the semantic imprecision introduced by machine translation. "Deplorable customer service" appearing explicitly as a trigram is noteworthy, this is a strong negative signal that survived translation intact and will be a useful feature.
"Euros per month" appearing (161 occurrences) is interesting, French reviewers frequently mention specific price points in euros, suggesting price anchoring is common. This could be exploited as a feature for pricing-related theme detection.

### Cross-Language Comparison: What This Means for our Models
The French n-grams are semantically denser and more precise, "rapport qualité prix", "sinistre non responsable", "prise en charge" are coherent insurance concepts. The English equivalents are either fragmented ("olive tree"), imprecise ("non responsible disaster"), or genericised ("good value money"). This definitively supports training our primary supervised models on French and reserving English only for GloVe initialisation.

--- CELL 41 (code) ---
# ─── Top words stratified by star rating (French & English) ──────────────────

# Columns to analyze
columns_to_analyze = ['avis_clean', 'avis_en_clean']

for col in columns_to_analyze:
    # Set labels for the report/titles
    lang_label = "French" if col == 'avis_clean' else "English"
    print(f"\n{'='*40}\n STRATIFIED ANALYSIS: {lang_label} ({col})\n{'='*40}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Filter data into Negative (<=2 stars) and Positive (>=4 stars) groups
    groups = [df[df['note'] <= 2], df[df['note'] >= 4]]
    colours = ['tomato', 'mediumseagreen']
    
    for star_group, ax, colour in zip(groups, axes, colours):
        # Join tokens for the specific column and star rating group
        tokens_group = ' '.join(star_group[col].dropna()).split()
        
        # Guard against empty groups
        if not tokens_group:
            ax.set_title(f"No samples for {col} in this rating group")
            continue
            
        top20 = Counter(tokens_group).most_common(20)
        words, freqs = zip(*top20)
        
        ax.barh(list(words)[::-1], list(freqs)[::-1], color=colour)
        
        rating_label = '1-2 Stars (Negative)' if colour == 'tomato' else '4-5 Stars (Positive)'
        ax.set_title(f'Top Words ({lang_label}), {rating_label}')
        ax.set_xlabel('Frequency')

    plt.tight_layout()
    # Save with unique filenames: ngrams_by_rating_avis_clean.png etc.
    plt.savefig(f'outputs/ngrams_by_rating_{col}.png', bbox_inches='tight')
    plt.show()


--- CELL 42 (markdown) ---
### The Positive Vocabulary is Tight and Evaluative

The positive French chart is dominated by "très" at ~6,000, by far the highest frequency token in either positive chart. This is the key finding: positive reviewers intensify their praise with adverbs. "Très" almost never appears as a standalone sentiment signal but as an amplifier, "très satisfait", "très rapide", "très bien", "très bon", "très simple", all of which appear further down the positive list. This confirms that degree adverbs are strong positive class features and should absolutely be preserved in our positive-class feature space, even though we flagged them earlier as stopword leakage candidates. We will therefore not remove "très" from our sentiment models.

"Rapide" and "simple" appearing in the positive top 20 are particularly insightful, speed of processing and simplicity of experience are the two operational qualities that most satisfy insurance customers. "Écoute" (attentiveness/listening) at the bottom of the positive list reinforces this: customers reward insurers who are responsive and easy to deal with.

"Merci" and "recommande" appearing explicitly in the positive chart are near-perfect sentiment proxies, a reviewer who writes "merci" is almost certainly leaving a positive review, and "recommande" maps directly to 4-5 stars in the vast majority of cases. These two tokens alone could form a high-precision positive heuristic rule.

### The Negative Vocabulary Negates the Temporal-Grievance Pattern

The negative French chart tells a very coherent story. "Mois" (~4,000) and "depuis" (~3,500) dominating the negative chart, "ans" present in the negative chart but not the positive one, meaning our earlier interpretation that short temporal units signal unresolved complaints and larger ones signal positive loyalty is probably not correct, the prensence of "ans" in the negative chart could be due to the fact that the loyalty of the customer is not rewarded and the customer feels betrayed.

"Sans" and "rien" both appearing in the negative top 20 confirms their value as negative sentiment markers despite being quasi-stopwords. These should be kept for sentiment models as discussed earlier. "Sinistre" appearing in the negative chart but not the positive one is also meaningful, claim events are inherently associated with negative experiences in this corpus, likely because customers who had smooth claims rarely feel the need to write about it in detail.

"Toujours" appearing in the negative list is subtle but important, in a negative review context "toujours" means "still" (as in "still no response", "still waiting") rather than "always" in a positive sense. This is a classic French polysemy trap and a reason why context matters more than unigrams for this project.

### English Charts Validate the French Findings

The English stratified charts largely mirror the French findings but with some additions worth noting.

On the negative side, "still", "without", "file", "even" all appearing confirms the proxy negative terms identified in the global frequency analysis.

On the positive side, "satisfied" at ~4,500 is the single clearest class separator across all four charts. "Fast", "simple", "easy", "clear" all appearing confirms the speed-and-simplicity driver of satisfaction identified in French. "Thank", "well", "recommend" mirror their French equivalents.

### Key Takeaway for Feature Engineering
These four charts give us a ready-made custom feature set. The cleanest sentiment separators, words that appear prominently in one chart but not the other, are:

Positive-only: très, rapide, simple, merci, recommande, satisfait, écoute, bien, bon
Negative-only: mois, depuis, sans, rien, sinistre, toujours, still, without, file, even

Any TF-IDF model that captures these unigrams well will have a strong baseline before we even reach bigrams. The fact that both languages produce the same underlying semantic clusters also means our French and English models should produce comparable results, a good sanity check to include in our model comparison section.

--- CELL 43 (code) ---
# ─── N-grams (Bigrams & Trigrams) Stratified by Rating & Language ─────────────

def get_ngram_freq(corpus_tokens: list, n: int = 2) -> Counter:
    """Compute n-gram frequency from a list of tokens."""
    from nltk import ngrams
    ng = [gram for token_list in corpus_tokens for gram in ngrams(token_list, n)]
    return Counter(ng)

def plot_ngrams(freq: Counter, n: int, col_name: str, rating_label: str, top_k: int = 20):
    label = 'Bigrams' if n == 2 else 'Trigrams'
    lang_suffix = "French" if col_name == 'avis_clean' else "English"
    
    top = pd.DataFrame(
        [(' '.join(g), c) for g, c in freq.most_common(top_k)],
        columns=['ngram', 'freq']
    )
    
    # Skip plotting if the results are empty
    if top.empty: return
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Use different colors for Pos vs Neg groups
    color = 'tomato' if 'Negative' in rating_label else 'mediumseagreen'
    
    ax.barh(top['ngram'][::-1], top['freq'][::-1], color=color)
    ax.set_title(f'Top {top_k} {label}, {lang_suffix} ({rating_label})')
    ax.set_xlabel('Frequency')
    
    plt.tight_layout()
    # Save with unique filename including language, rating group, and ngram type
    rating_fn = rating_label.split()[0].lower() # 'negative' or 'positive'
    plt.savefig(f'outputs/ngrams_{label.lower()}_{col_name}_{rating_fn}.png', bbox_inches='tight')
    plt.show()

# Define columns and rating groups to analyze
columns = ['avis_clean', 'avis_en_clean']
rating_groups = [
    (df[df['note'] <= 2], "1-2 Stars (Negative)"),
    (df[df['note'] >= 4], "4-5 Stars (Positive)")
]

for col in columns:
    for group_df, group_label in rating_groups:
        # Pre-tokenize specific group
        tokenised = [sent.split() for sent in group_df[col].dropna() if sent.strip()]
        
        if not tokenised: continue

        # Compute frequencies
        bigram_freq  = get_ngram_freq(tokenised, 2)
        trigram_freq = get_ngram_freq(tokenised, 3)

        # Plot only (no print/display)
        plot_ngrams(bigram_freq, 2, col, group_label)
        plot_ngrams(trigram_freq, 3, col, group_label)


--- CELL 44 (markdown) ---
### "Depuis plus ans" is Definitively Negative
The question we left open is now answered. "Depuis plus ans" appears 260 times in the negative trigrams (rank 1) versus only 64 times in the positive trigrams (rank 14). The ratio is roughly 4:1 negative. Our earlier hypothesis that it was a positive loyalty marker was wrong, the correct interpretation is the betrayed loyalty framing: long-tenure customers invoking their years of loyalty to emphasise grievance. "Client depuis ans" appearing in the negative trigrams at rank 11 further confirms this. The "ans" unigram being exclusively in the negative top 20 is now fully explained.

### The Negative French Trigrams Tell a Complete Story
The negative trigram list is remarkably coherent and reads almost like a taxonomy of insurance complaints:

Claims disputes: "accident non responsable" (126), "sinistre non responsable" (124), "sans aucun sinistre" (67), "assuré tout risque" (54), customers disputing fault attribution or feeling let down despite having comprehensive coverage
Temporal grievance: "depuis plus ans" (260), "mois plus tard" (104), "depuis plusieurs années" (71), "cela fait mois" (68), "jours plus tard" (67), "depuis plus mois" (65), the unresolved waiting pattern across multiple time scales
Explicit rejection language: "fuyez cette assurance" (55), "déconseille fortement cette" (54), "fortement cette assurance" (47), "éviter tout prix" (46), "aller voir ailleurs" (45), these are near-perfect negative sentiment classifiers with essentially zero ambiguity. Any review containing "fuyez" or "éviter tout prix" is a 1-star review with very high probability
Service failure: "service client déplorable" (52), "aucune réponse" (341 as bigram), direct service quality complaints


### The Positive French Trigrams are Formulaic and Repetitive
The positive trigrams cluster tightly around a small set of templates: "rapport qualité prix" (327), "satisfait service prix" (231), "bon rapport qualité" (157), "très satisfait service" (145). These are essentially variations of the same two sentences, "good value for money" and "satisfied with service and price". This confirms the earlier observation that positive reviews are lexically formulaic while negative reviews are diverse and specific. The practical implication is that our positive class will be easier to classify with TF-IDF because the signal is concentrated, while negative reviews will require richer representations to capture the full range of complaint types.

### English Negative Trigrams Add Unique Signal
Several English negative trigrams have no clean French equivalent and add genuine value:

"Still nothing" (170 as bigram), "want know anything" (30), "want hear anything" (23), these capture the insurer's lack of engagement, a concept expressed more idiomatically in English translation
"Impossible reach phone" (27), a specific channel failure complaint about phone accessibility
"Run away absolutely" (36), "strongly advise insurance" (37), explicit avoidance recommendations
"Incompetent customer service" (32), "deplorable customer service" (47), direct quality judgements that are cleaner in English than their French equivalents

"Years without claim" (24) is particularly interesting, it's a customer explicitly stating they never filed a claim yet still had a bad experience, likely around cancellation or pricing, which is a specific and legitimate grievance type worth flagging.

### The "Olive Tree" Problem Quantified
"Olive tree" appears 193 times as a positive bigram and "recommend olive tree" / "insurance olive tree" appear in the positive trigrams. This is entirely L'Olivier Assurance being mistranslated. It's not a catastrophic problem for sentiment modelling since the mistranslation is consistent, the model will learn "olive tree" as a positive feature, but it means any cross-corpus or cross-domain evaluation would fail on this token. Worth explicitly noting in our results section as a translation artefact.

### "Mutual Insurance Company" Appearing in Both Positive and Negative
"Mutual insurance company" ranks 1st in negative trigrams (168) and 10th in positive (57). This is a translation of "mutuelle", a specific French insurance structure. Its appearance in both lists confirms it's a product-type label rather than a sentiment signal, and is a good example of a term that should be treated as a categorical feature rather than a sentiment feature in our models.

--- CELL 45 (markdown) ---
### 3.4 Spelling Correction

Insurance reviews often contain informal language and typos. We use `pyspellchecker` which supports French. We create a new column `avis_corrected` with corrections applied.

--- CELL 46 (code) ---
# ─── Spelling correction ─────────────────────────────────────────────────────
from spellchecker import SpellChecker

spell_fr = SpellChecker(language='fr')
spell_en = SpellChecker(language='en')

def correct_spelling(text: str, lang: str = 'fr') -> str:
    """Correct spelling mistakes word by word."""
    if not isinstance(text, str) or not text.strip():
        return text
    spell = spell_fr if lang == 'fr' else spell_en
    words = text.split()
    corrected = []
    for word in words:
        # Only correct alphabetic tokens; leave numbers/proper nouns
        if word.isalpha() and len(word) > 3:
            correction = spell.correction(word)
            corrected.append(correction if correction else word)
        else:
            corrected.append(word)
    return ' '.join(corrected)

# Apply on a sample to demo
SAMPLE_SIZE = min(2000, len(df))
print(f'Applying spelling correction to {SAMPLE_SIZE} reviews...')
sample_idx = df.sample(SAMPLE_SIZE, random_state=42).index

df['avis_corrected'] = df['avis_light'].copy()  # start with light-clean text
df.loc[sample_idx, 'avis_corrected'] = df.loc[sample_idx, 'avis_light'].apply(
    lambda x: correct_spelling(x, lang='fr')
)

# 1. Filter for reviews where a correction was actually made
changed_mask = df.loc[sample_idx, 'avis_light'] != df.loc[sample_idx, 'avis_corrected']
examples = df.loc[sample_idx][changed_mask].head(5)

print('\n' + '='*80)
print('   SPELLING CORRECTION EXAMPLES (ONLY MODIFIED REVIEWS)')
print('='*80)

if examples.empty:
    print("No spelling corrections were made in the sample.")
else:
    for i, (idx, row) in enumerate(examples.iterrows(), 1):
        print(f"\nEXAMPLE #{i} (Index: {idx})")
        print(f"{'-'*30}")
        print(f"ORIGINAL RAW REVIEW:\n{row['avis']}")
        print(f"\nBEFORE CORRECTION (Light Cleaned):\n{row['avis_light']}")
        print(f"\nAFTER CORRECTION:\n{row['avis_corrected']}")
        print(f"\n{'-'*80}")



--- CELL 47 (code) ---
# ─── Measure correction rate ─────────────────────────────────────────────────
def count_changes(original: str, corrected: str) -> int:
    orig_words = str(original).split()
    corr_words = str(corrected).split()
    return sum(1 for a, b in zip(orig_words, corr_words) if a != b)

df.loc[sample_idx, 'n_corrections'] = df.loc[sample_idx].apply(
    lambda row: count_changes(row['avis_light'], row['avis_corrected']), axis=1
)
avg_corr = df.loc[sample_idx, 'n_corrections'].mean()
print(f'Average corrections per review: {avg_corr:.2f}')

fig, ax = plt.subplots(figsize=(8, 4))
df.loc[sample_idx, 'n_corrections'].value_counts().sort_index().head(15).plot.bar(ax=ax, color='orchid')
ax.set_title('Number of Spelling Corrections per Review')
ax.set_xlabel('# Corrections')
plt.tight_layout()
plt.savefig('outputs/spelling_corrections.png', bbox_inches='tight')
plt.show()

--- CELL 48 (markdown) ---
### Spelling Correction Distribution

The histogram shows a healthy long-tail distribution. The majority of reviews (~930) received zero corrections, meaning the corpus is relatively well-written overall, these are considered reviews on a structured platform, not SMS-style messages. About 470 received 1 correction and 215 received 2, with frequency dropping sharply beyond that. The tail extends to 16 corrections but those are rare edge cases, likely very long or very poorly written reviews.

### What the Corrections Actually Look Like
Examining only the reviews where changes were made reveals a mixed picture with three distinct outcome types.
Genuine successes are present and meaningful. "tres" → "très", "interressant" → "intéressant", "preferentiel" → "préférentiel" are exactly the corrections the tool should be making, common French words where the writer omitted accents, a very frequent pattern in informal online writing. These are unambiguous improvements that will help Word2Vec and TF-IDF models by consolidating unaccented and accented variants into the same token.
Harmful corrections are equally present however. "parceque" → "parcelle" is a damaging substitution, "parceque" is an informal contracted spelling of "parce que" (because), common in online French writing, which the corrector replaces with "parcelle" (plot of land), a semantically nonsensical substitution that would actively confuse downstream models. More critically, "april" → "avril" silently replaces the insurer name APRIL, one of the top 10 insurers in the corpus, with the French word for the month of April. This is a named entity recognition failure that would corrupt any insurer-level analysis.
Trivial changes make up the remainder. "aujourd hui" → "aujourd' hui" is an artefact of apostrophe stripping in avis_light and adds no value. "bcp" (informal abbreviation of "beaucoup") was left uncorrected, confirming the tool cannot handle informal abbreviations which are common throughout the corpus.

### Overall Assessment
The corrector has a precision problem more than a recall problem, when it does intervene, it is wrong or harmful roughly as often as it is right. The dominant zero bar in the histogram is therefore not purely a sign of a clean corpus but also reflects the tool largely operating as a no-op, which is arguably preferable to the alternative of it actively corrupting named entities and informal spellings at scale.
The honest conclusion is that pyspellchecker is too blunt an instrument for this corpus for two compounding reasons: French insurance reviews contain domain-specific proper nouns that the general dictionary does not know, and informal contracted spellings fall entirely outside its correction model. The accent restoration case is where it genuinely adds value, and a more targeted approach, specifically correcting only unaccented variants of known French common words while maintaining a protected vocabulary of insurer names, would be more defensible. A more robust alternative altogether would be language-tool-python, which handles French morphology and grammar rather than relying on simple dictionary lookup.

--- CELL 49 (code) ---
# ─── Improved Spelling Correction ────────────────────────────────────────────
from spellchecker import SpellChecker
import re

spell_fr = SpellChecker(language='fr')

# ── 1. Protected vocabulary: insurer names and domain terms ──────────────────
# These will never be corrected regardless of dictionary lookup
INSURER_NAMES = {
    'april', 'maif', 'macif', 'maaf', 'gmf', 'axa', 'allianz', 'aviva',
    'santiane', 'neoliane', 'amv', 'matmut', 'groupama', 'covea', 'pacifica',
    'swisslife', 'generali', 'metlife', 'cardif', 'harmonie', 'malakoff',
    'mercer', 'interiale', 'suravenir', 'hiscox', 'magnolia', 'mgen',
    'olivier', 'zenup', 'leocare', 'direct', 'eurofil', 'lelynx'
}

INSURANCE_DOMAIN_TERMS = {
    'sinistre', 'assureur', 'assuré', 'assurée', 'cotisation', 'résiliation',
    'résiliée', 'résilier', 'remboursement', 'rembourser', 'franchise',
    'indemnisation', 'indemnité', 'préjudice', 'contentieux', 'adhésion',
    'souscription', 'souscrire', 'garantie', 'couverture', 'prélèvement',
    'assurance', 'contrat', 'mutuelle', 'prévoyance', 'habitation',
    'responsabilité', 'tiers', 'malus', 'bonus', 'prime', 'tarif',
    'expertise', 'expert', 'constat', 'déclaration', 'dommage'
}

# Common informal contractions that the corrector would mangle
INFORMAL_CONTRACTIONS = {
    'parceque', 'parceke', 'bcp', 'tjrs', 'tjs', 'svp', 'stp',
    'rdv', 'tel', 'msg', 'info', 'vs', 'pr', 'pck', 'pk', 'pcq'
}

PROTECTED_VOCAB = INSURER_NAMES | INSURANCE_DOMAIN_TERMS | INFORMAL_CONTRACTIONS

# Add protected vocab to spell checker so it never flags them as errors
spell_fr.word_frequency.load_words(PROTECTED_VOCAB)

# ── 2. Accent restoration map ─────────────────────────────────────────────────
# The one case where the corrector reliably adds value: unaccented → accented
# We handle the most common cases explicitly rather than relying on the dictionary
ACCENT_MAP = {
    'tres':          'très',
    'etre':          'être',
    'meme':          'même',
    'votre':         'votre',   # already correct, no change
    'notre':         'notre',
    'autre':         'autre',
    'apres':         'après',
    'deja':          'déjà',
    'aussi':         'aussi',
    'toujours':      'toujours',
    'interressant':  'intéressant',
    'interessant':   'intéressant',
    'preferentiel':  'préférentiel',
    'prefere':       'préféré',
    'preferer':      'préférer',
    'efficacement':  'efficacement',
    'telephonique':  'téléphonique',
    'telephone':     'téléphone',
    'immediat':      'immédiat',
    'immediatement': 'immédiatement',
    'ecoute':        'écoute',
    'ecoutent':      'écoutent',
    'reponse':       'réponse',
    'repondre':      'répondre',
    'remuneration':  'rémunération',
    'probleme':      'problème',
    'systeme':       'système',
    'resiliation':   'résiliation',
    'resilier':      'résilier',
    'assurance':     'assurance',
}

def correct_spelling_improved(text: str) -> str:
    """
    Improved spelling correction with three safeguards:
    1. Protected vocabulary (insurer names, domain terms, informal contractions)
       are never modified
    2. Explicit accent restoration map handles the highest-value correction case
    3. pyspellchecker only applied to remaining unprotected, non-accent words
       as a last resort, with a confidence check
    """
    if not isinstance(text, str) or not text.strip():
        return text

    words = text.split()
    corrected = []

    for word in words:
        word_lower = word.lower()

        # Guard 1: skip short tokens and non-alphabetic
        if not word.isalpha() or len(word) <= 3:
            corrected.append(word)
            continue

        # Guard 2: never touch protected vocabulary
        if word_lower in PROTECTED_VOCAB:
            corrected.append(word)
            continue

        # Guard 3: apply accent map first (highest precision)
        if word_lower in ACCENT_MAP:
            corrected.append(ACCENT_MAP[word_lower])
            continue

        # Guard 4: pyspellchecker as last resort
        # Only apply if the word is flagged as misspelled AND
        # the suggested correction shares at least 60% of characters
        # (prevents wild substitutions like parceque → parcelle)
        if word_lower in spell_fr:
            # Word is known to dictionary, leave it
            corrected.append(word)
        else:
            suggestion = spell_fr.correction(word_lower)
            if suggestion and suggestion != word_lower:
                # Similarity check: reject if correction is too different
                shorter = min(len(word_lower), len(suggestion))
                longer  = max(len(word_lower), len(suggestion))
                # Count matching characters at same position
                matches = sum(a == b for a, b in zip(word_lower, suggestion))
                similarity = matches / longer
                if similarity >= 0.6:
                    corrected.append(suggestion)
                else:
                    # Correction too dissimilar, leave original
                    corrected.append(word)
            else:
                corrected.append(word)

    return ' '.join(corrected)


# ── 3. Apply correction ───────────────────────────────────────────────────────
SAMPLE_SIZE = min(2000, len(df))
print(f'Applying improved spelling correction to {SAMPLE_SIZE} reviews...')
sample_idx = df.sample(SAMPLE_SIZE, random_state=42).index

df['avis_corrected'] = df['avis'].copy()
df.loc[sample_idx, 'avis_corrected'] = df.loc[sample_idx, 'avis'].apply(
    correct_spelling_improved
)

# ── 4. Evaluate: show only modified reviews ───────────────────────────────────
changed_mask = df.loc[sample_idx, 'avis'] != df.loc[sample_idx, 'avis_corrected']
n_changed = changed_mask.sum()
print(f'Reviews modified: {n_changed} / {SAMPLE_SIZE} ({n_changed/SAMPLE_SIZE*100:.1f}%)')

examples = df.loc[sample_idx][changed_mask].head(5)

print('\n' + '='*80)
print('   SPELLING CORRECTION EXAMPLES (ONLY MODIFIED REVIEWS)')
print('='*80)

if examples.empty:
    print('No spelling corrections were made in the sample.')
else:
    for i, (idx, row) in enumerate(examples.iterrows(), 1):
        # Show word-level diff
        before_words = row['avis'].split()
        after_words  = row['avis_corrected'].split()
        changes = [
            f"  '{b}' → '{a}'"
            for b, a in zip(before_words, after_words)
            if b != a
        ]
        print(f'\nEXAMPLE #{i} (Index: {idx})')
        print('-' * 30)
        print(f'ORIGINAL:  {row["avis"]}')
        print(f'AFTER:     {row["avis_corrected"]}')
        print(f'CHANGES:')
        for c in changes:
            print(c)
        print('-' * 80)

# ── 5. Correction rate distribution ──────────────────────────────────────────
def count_word_changes(before: str, after: str) -> int:
    b_words = str(before).split()
    a_words = str(after).split()
    return sum(1 for b, a in zip(b_words, a_words) if b != a)

df.loc[sample_idx, 'n_corrections'] = df.loc[sample_idx].apply(
    lambda row: count_word_changes(row['avis'], row['avis_corrected']), axis=1
)

fig, ax = plt.subplots(figsize=(10, 4))
df.loc[sample_idx, 'n_corrections'].value_counts().sort_index().head(15).plot.bar(
    ax=ax, color='orchid', edgecolor='white'
)
ax.set_title('Number of Spelling Corrections per Review (Improved Corrector)')
ax.set_xlabel('# Corrections')
ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig('outputs/spelling_corrections_improved.png', bbox_inches='tight')
plt.show()

print('\nCorrection summary:')
print(df.loc[sample_idx, 'n_corrections'].describe().round(2))

--- CELL 50 (code) ---
# ─── Improved Spelling Correction with LanguageTool ──────────────────────────
import language_tool_python
import difflib
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ── 1. Initialise French LanguageTool ─────────────────────────────────────────
print('Initialising LanguageTool (French)...')
tool = language_tool_python.LanguageTool('fr')

# Disable noisy categories directly on the instance after initialisation
tool.disabled_categories = {
    'STYLE',
    'PUNCTUATION', 
    'TYPOGRAPHY',
    'CASING',
    'REDUNDANCY',
    'MISC'
}
print(f'✅ LanguageTool ready')
print(f'   Disabled categories: {tool.disabled_categories}')

# ── 2. Protected vocabulary ───────────────────────────────────────────────────
PROTECTED_VOCAB = {
    # Insurer names
    'april', 'maif', 'macif', 'maaf', 'gmf', 'axa', 'allianz',
    'santiane', 'neoliane', 'amv', 'matmut', 'groupama', 'pacifica',
    'swisslife', 'generali', 'metlife', 'cardif', 'harmonie', 'malakoff',
    'mercer', 'interiale', 'suravenir', 'hiscox', 'magnolia', 'mgen',
    'olivier', 'zenup', 'leocare', 'eurofil', 'lelynx', 'aviva',
    # Informal abbreviations, leave as-is
    'bcp', 'tjrs', 'tjs', 'svp', 'stp', 'rdv', 'vs', 'pr', 'pck',
}

DISABLED_RULES = {
    'UPPERCASE_SENTENCE_START',
    'FRENCH_WHITESPACE',
    'COMMA_PARENTHESIS_WHITESPACE',
    'DOUBLE_PUNCTUATION',
    'UNPAIRED_BRACKETS',
    'APOS_TYP',
}

# ── 3. Core correction function ───────────────────────────────────────────────
def correct_with_languagetool(text: str, max_length: int = 300) -> str:
    """
    Correct spelling using LanguageTool French.
    max_length reduced to 300 for speed, covers the vast majority of reviews.
    """
    if not isinstance(text, str) or not text.strip():
        return text

    text_to_correct = text[:max_length]

    try:
        matches = tool.check(text_to_correct)
    except Exception:
        return text

    matches = sorted(matches, key=lambda m: m.offset, reverse=True)

    corrected = text_to_correct
    for match in matches:

        if match.rule_id in DISABLED_RULES:
            continue

        if not match.replacements:
            continue

        original_word = corrected[match.offset: match.offset + match.error_length]
        suggested     = match.replacements[0]

        if original_word.lower() in PROTECTED_VOCAB:
            continue

        if suggested[0].isupper() and not original_word[0].isupper():
            continue

        if len(suggested.split()) > 2:
            continue

        corrected = (
            corrected[:match.offset]
            + suggested
            + corrected[match.offset + match.error_length:]
        )

    if len(text) > max_length:
        corrected = corrected + text[max_length:]

    return corrected

# ── 4. Parallel application ───────────────────────────────────────────────────
SAMPLE_SIZE = min(2000, len(df))
sample_idx  = df.sample(SAMPLE_SIZE, random_state=42).index
texts_to_correct = df.loc[sample_idx, 'avis'].tolist()

def correct_batch(texts: list, max_workers: int = 4) -> list:
    results = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(correct_with_languagetool, text): idx
            for idx, text in enumerate(texts)
        }
        for future in tqdm(
            as_completed(future_to_idx),
            total=len(texts),
            desc='Correcting'
        ):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = texts[idx]
    return results

print(f'Applying LanguageTool correction to {SAMPLE_SIZE} reviews (parallel, 4 workers)...')
df['avis_corrected'] = df['avis'].copy()
corrected_texts = correct_batch(texts_to_correct, max_workers=4)
df.loc[sample_idx, 'avis_corrected'] = corrected_texts

# ── 5. Change detection with difflib ─────────────────────────────────────────
def get_word_changes(before: str, after: str) -> list:
    """
    Use difflib SequenceMatcher to correctly identify changed tokens
    even when insertions/deletions shift word positions.
    Returns list of (before_chunk, after_chunk) tuples for actual changes.
    """
    before_words = str(before).split()
    after_words  = str(after).split()
    matcher      = difflib.SequenceMatcher(None, before_words, after_words)
    changes      = []

    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        if opcode == 'replace':
            changes.append((
                ' '.join(before_words[i1:i2]),
                ' '.join(after_words[j1:j2])
            ))
        elif opcode == 'delete':
            changes.append((' '.join(before_words[i1:i2]), '[DELETED]'))
        elif opcode == 'insert':
            changes.append(('[INSERTED]', ' '.join(after_words[j1:j2])))

    return changes

def count_word_changes_fixed(before: str, after: str) -> int:
    return len(get_word_changes(before, after))

# ── 6. Display examples ───────────────────────────────────────────────────────
changed_mask = df.loc[sample_idx, 'avis'] != df.loc[sample_idx, 'avis_corrected']
n_changed    = changed_mask.sum()
print(f'\nReviews modified: {n_changed} / {SAMPLE_SIZE} ({n_changed/SAMPLE_SIZE*100:.1f}%)')

examples = df.loc[sample_idx][changed_mask].head(5)

print('\n' + '='*80)
print('   LANGUAGETOOL CORRECTION EXAMPLES (ONLY MODIFIED REVIEWS)')
print('='*80)

if examples.empty:
    print('No corrections were made in the sample.')
else:
    for i, (idx, row) in enumerate(examples.iterrows(), 1):
        changes = get_word_changes(row['avis'], row['avis_corrected'])
        print(f'\nEXAMPLE #{i} (Index: {idx})')
        print('-' * 30)

        print(f'BEFORE:    {row["avis"]}')
        print(f'AFTER:     {row["avis_corrected"]}')
        print(f'CHANGES ({len(changes)}):')
        for before_chunk, after_chunk in changes:
            print(f"  '{before_chunk}' → '{after_chunk}'")
        print('-' * 80)

# ── 7. Correction rate distribution ──────────────────────────────────────────
df.loc[sample_idx, 'n_corrections'] = df.loc[sample_idx].apply(
    lambda row: count_word_changes_fixed(
        row['avis'], row['avis_corrected']
    ), axis=1
)

fig, ax = plt.subplots(figsize=(10, 4))
df.loc[sample_idx, 'n_corrections'].value_counts().sort_index().head(20).plot.bar(
    ax=ax, color='mediumseagreen', edgecolor='white'
)
ax.set_title('Number of Corrections per Review (LanguageTool)')
ax.set_xlabel('# Corrections')
ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig('outputs/spelling_corrections_languagetool.png', bbox_inches='tight')
plt.show()

print('\nCorrection summary:')
print(df.loc[sample_idx, 'n_corrections'].describe().round(2))

# ── 8. Cleanup ────────────────────────────────────────────────────────────────
tool.close()
print('LanguageTool server closed.')

--- CELL 51 (markdown) ---
### Spelling Correction Distribution
The histogram now shows the expected distribution for a structured review platform, a clean geometric decay peaking at 1 correction per review (~440 reviews), dropping steadily through 2, 3, 4 corrections, with a long tail that cuts off at 14. This is a credible and interpretable distribution. The mean of 2.45 corrections per review with a std of 2.26 indicates most reviews need only minor corrections, while the max of 14 represents the most informally written reviews in the sample, well within the plausible range for user-generated content.

### Why We Correct avis and Not avis_light
The decision to apply spelling correction to the raw avis rather than the normalised avis_light is both principled and practically validated by the examples. LanguageTool is a grammar-aware tool that analyses sentence structure to identify errors, it needs the full original text with punctuation, capitalisation, and apostrophes intact to function correctly. When we tried previously to correct avis_light, the normalisation pipeline had already stripped apostrophes, merged sentences by removing punctuation, and removed numbers, creating artefacts like "aujourd hui" and "plus de ans" that were not reviewer errors but pipeline artefacts. LanguageTool was then spending most of its effort correcting damage we introduced ourselves rather than genuine mistakes. Correcting on avis means LanguageTool only ever encounters genuine reviewer errors in their proper grammatical context, which is why Example 5 correctly distinguishes "a" (verb) from "à" (preposition), a disambiguation that requires surrounding context and was impossible on the stripped text. The corrected avis_corrected then flows into normalisation to produce avis_corrected_clean and avis_corrected_light, meaning every downstream model benefits from properly corrected source text.

### Quality Assessment of the Examples
The five examples show a strong improvement over all previous implementations. Example 1 catches a genuine conjugation error ("J'attend" → "J'attends") that neither pyspellchecker nor the accent map could identify since it requires grammatical context. Example 4 correctly fixes number agreement ("multiples" → "multiple"). Example 5 restores the preposition accent twice ("a" → "à") with perfect precision, correctly identifying both instances as prepositions from context rather than the verb "a". Example 2 correctly splits "parceque" into "parce que" rather than mangling it into "parcelle" as the original implementation did.
The one remaining false positive is "perte" → "parte" in Example 2, where a perfectly correct French word ("perte de temps", waste of time) is incorrectly flagged. This is a low-frequency failure mode inherent to any automated correction system and acceptable at this scale.

### Is This Spelling Correction Good Enough for our Project?
For the purposes of our project, yes. No automated spelling corrector will achieve perfection on informal, domain-specific user-generated text, the question is whether the corrections add more value than they remove. Based on the examples and the distribution, the answer is clearly yes: the corrector fixes genuine errors (conjugation mistakes, missing accents, informal contractions like "parceque") far more often than it introduces new ones. The false positive rate from the five examples is one out of nine total changes, or roughly 11%, well within acceptable bounds for a preprocessing step whose role is to improve downstream model input quality rather than produce publication-ready text. The protected vocabulary prevents the most damaging class of errors (insurer name corruption) entirely, and the disabled categories prevent over-correction of intentional informal style. The result is a corrected corpus that will produce cleaner token distributions, better Word2Vec embeddings, and more reliable TF-IDF features without meaningfully distorting the sentiment signal.

--- CELL 52 (markdown) ---
### Applying the correction to the entire dataset

--- CELL 53 (code) ---

# ── 1. Initialise French LanguageTool ─────────────────────────────────────────
print('Initialising LanguageTool (French)...')
tool = language_tool_python.LanguageTool('fr')

# Disable noisy categories directly on the instance after initialisation
tool.disabled_categories = {
    'STYLE',
    'PUNCTUATION', 
    'TYPOGRAPHY',
    'CASING',
    'REDUNDANCY',
    'MISC'
}
print(f'✅ LanguageTool ready')
print(f'   Disabled categories: {tool.disabled_categories}')

# ── 2. Protected vocabulary ───────────────────────────────────────────────────
PROTECTED_VOCAB = {
    # Insurer names
    'april', 'maif', 'macif', 'maaf', 'gmf', 'axa', 'allianz',
    'santiane', 'neoliane', 'amv', 'matmut', 'groupama', 'pacifica',
    'swisslife', 'generali', 'metlife', 'cardif', 'harmonie', 'malakoff',
    'mercer', 'interiale', 'suravenir', 'hiscox', 'magnolia', 'mgen',
    'olivier', 'zenup', 'leocare', 'eurofil', 'lelynx', 'aviva',
    # Informal abbreviations, leave as-is
    'bcp', 'tjrs', 'tjs', 'svp', 'stp', 'rdv', 'vs', 'pr', 'pck',
}

DISABLED_RULES = {
    'UPPERCASE_SENTENCE_START',
    'FRENCH_WHITESPACE',
    'COMMA_PARENTHESIS_WHITESPACE',
    'DOUBLE_PUNCTUATION',
    'UNPAIRED_BRACKETS',
    'APOS_TYP',
}

# ── 3. Core correction function ───────────────────────────────────────────────
def correct_with_languagetool(text: str, max_length: int = 300) -> str:
    """
    Correct spelling using LanguageTool French.
    max_length reduced to 300 for speed, covers the vast majority of reviews.
    """
    if not isinstance(text, str) or not text.strip():
        return text

    text_to_correct = text[:max_length]

    try:
        matches = tool.check(text_to_correct)
    except Exception:
        return text

    matches = sorted(matches, key=lambda m: m.offset, reverse=True)

    corrected = text_to_correct
    for match in matches:

        if match.rule_id in DISABLED_RULES:
            continue

        if not match.replacements:
            continue

        original_word = corrected[match.offset: match.offset + match.error_length]
        suggested     = match.replacements[0]

        if original_word.lower() in PROTECTED_VOCAB:
            continue

        if suggested[0].isupper() and not original_word[0].isupper():
            continue

        if len(suggested.split()) > 2:
            continue

        corrected = (
            corrected[:match.offset]
            + suggested
            + corrected[match.offset + match.error_length:]
        )

    if len(text) > max_length:
        corrected = corrected + text[max_length:]

    return corrected

# ── 4. Parallel application ───────────────────────────────────────────────────
texts_to_correct = df['avis'].tolist()

def correct_batch(texts: list, max_workers: int = 4) -> list:
    results = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(correct_with_languagetool, text): idx
            for idx, text in enumerate(texts)
        }
        for future in tqdm(
            as_completed(future_to_idx),
            total=len(texts),
            desc='Correcting'
        ):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = texts[idx]
    return results

print(f'Applying LanguageTool correction to {len(df):,} reviews (parallel, 4 workers)...')
df['avis_corrected'] = df['avis'].copy()
corrected_texts = correct_batch(texts_to_correct, max_workers=4)
df['avis_corrected'] = corrected_texts

# ── 5. Change detection with difflib ─────────────────────────────────────────
def get_word_changes(before: str, after: str) -> list:
    """
    Use difflib SequenceMatcher to correctly identify changed tokens
    even when insertions/deletions shift word positions.
    Returns list of (before_chunk, after_chunk) tuples for actual changes.
    """
    before_words = str(before).split()
    after_words  = str(after).split()
    matcher      = difflib.SequenceMatcher(None, before_words, after_words)
    changes      = []

    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        if opcode == 'replace':
            changes.append((
                ' '.join(before_words[i1:i2]),
                ' '.join(after_words[j1:j2])
            ))
        elif opcode == 'delete':
            changes.append((' '.join(before_words[i1:i2]), '[DELETED]'))
        elif opcode == 'insert':
            changes.append(('[INSERTED]', ' '.join(after_words[j1:j2])))

    return changes

def count_word_changes_fixed(before: str, after: str) -> int:
    return len(get_word_changes(before, after))

# ── 6. Display examples ───────────────────────────────────────────────────────
changed_mask = df['avis'] != df['avis_corrected']
n_changed    = changed_mask.sum()
print(f'\nReviews modified: {n_changed} / {len(df):,} ({n_changed/len(df)*100:.1f}%)')

examples = df[changed_mask].head(5)

print('\n' + '='*80)
print('   LANGUAGETOOL CORRECTION EXAMPLES (ONLY MODIFIED REVIEWS)')
print('='*80)

if examples.empty:
    print('No corrections were made.')
else:
    for i, (idx, row) in enumerate(examples.iterrows(), 1):
        changes = get_word_changes(row['avis'], row['avis_corrected'])
        print(f'\nEXAMPLE #{i} (Index: {idx})')
        print('-' * 30)
        print(f'BEFORE:    {row["avis"]}')
        print(f'AFTER:     {row["avis_corrected"]}')
        print(f'CHANGES ({len(changes)}):')
        for before_chunk, after_chunk in changes:
            print(f"  '{before_chunk}' → '{after_chunk}'")
        print('-' * 80)

# ── 7. Correction rate distribution ──────────────────────────────────────────
df['n_corrections'] = df.apply(
    lambda row: count_word_changes_fixed(
        row['avis'], row['avis_corrected']
    ), axis=1
)

fig, ax = plt.subplots(figsize=(10, 4))
df['n_corrections'].value_counts().sort_index().head(20).plot.bar(
    ax=ax, color='mediumseagreen', edgecolor='white'
)
ax.set_title('Number of Corrections per Review (LanguageTool)')
ax.set_xlabel('# Corrections')
ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig('outputs/spelling_corrections_languagetool.png', bbox_inches='tight')
plt.show()

print('\nCorrection summary:')
print(df['n_corrections'].describe().round(2))

# ── 8. Cleanup ────────────────────────────────────────────────────────────────
tool.close()
print('LanguageTool server closed.')

--- CELL 54 (code) ---
df.to_parquet('outputs/reviews_corrected.parquet', index=False)
print(f'✅ Saved → outputs/reviews_corrected.parquet  ({len(df):,} rows)')
df.head()

--- CELL 55 (code) ---
# ─── Derive normalised columns from corrected raw text ────────────────────────
# Now that avis_corrected contains spelling-corrected raw text,
# we re-apply the same normalisation pipeline to produce clean variants.
# This ensures all downstream models benefit from corrected source text.

print('Normalising corrected text...')

# For TF-IDF, Word2Vec, topic modelling, no apostrophes, stopwords removed
df['avis_corrected_clean'] = df['avis_corrected'].apply(clean_text)

# For BERT, LSTM, sentiment models, apostrophes preserved, stopwords kept
df['avis_corrected_light'] = df['avis_corrected'].apply(clean_text_keep_stops)

print('Done!')
print(f'Rows with corrected text: {df["avis_corrected"].notna().sum():,}')
print(f'Rows where correction changed the text: '
      f'{(df["avis_corrected"] != df["avis"]).sum():,}')

# Verify the pipeline worked correctly with a sample
display(
    df[['avis', 'avis_corrected', 'avis_corrected_clean', 'avis_corrected_light']]
    .loc[df['avis_corrected'] != df['avis']]  # only show modified rows
    .head(5)
)

--- CELL 56 (code) ---
missing = df.isnull().sum()
print(missing[missing > 0])

--- CELL 57 (code) ---
# ─── Drop old correction columns ───────────────────────────────────────
cols_to_drop = [col for col in ['avis_cor', 'avis_cor_en'] if col in df.columns]
df.drop(columns=cols_to_drop, inplace=True)

--- CELL 58 (code) ---
df.head()

--- CELL 59 (markdown) ---
We now have completed the correction and normalisation of the reviews, we have not corrected the english reviews for the following reasons:

Our decision was to use French as the primary modelling language and English only for GloVe. The GloVe models are robust to minor spelling errors in the input, a misspelled word simply gets a lower-quality or zero embedding rather than corrupting the model the way it might corrupt a TF-IDF vocabulary.

The translations already contain their own errors. avis_en is machine-translated text, meaning it already has translation artefacts like "olive tree" for L'Olivier and inconsistent renderings of "sinistre". Running a spelling corrector on top of a machine translation would be correcting a secondary source that is already an imperfect representation of the original, the errors in avis_en are mostly semantic and structural rather than spelling-based, and LanguageTool cannot fix those.

The correct approach would be to translate from avis_corrected. If clean English text matters for our downstream tasks, the right pipeline is to correct the French first (which we did), then translate avis_corrected to English rather than trying to correct the existing avis_en. This would give us avis_corrected_en derived from properly corrected source text. However given that English is only needed for GloVe in our current project, the effort is not justified by the marginal quality gain.

--- CELL 60 (markdown) ---
<a id='4'></a>
## 4. Section 2, Summary, Translation & Generation (2 points)

We use the **Qwen2.5-1.5B-Instruct** LLM for:
1. **Translation**, translate French reviews to English where `avis_en` is missing
2. **Summarisation**, generate concise summaries of groups of reviews per insurer
3. **Generation**, produce synthetic review examples for data augmentation

We then export a clean master CSV with all processed columns.

--- CELL 61 (code) ---
# ─── Load Qwen2.5-1.5B-Instruct ──────────────────────────────────────────────
MODEL_NAME = 'Qwen/Qwen2.5-1.5B-Instruct'

print(f'Loading {MODEL_NAME}...')
qwen_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
qwen_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16 if DEVICE == 'cuda' else torch.float32,
    device_map='auto'
)
qwen_model.eval()
print('✅ Qwen2.5-1.5B-Instruct loaded')

--- CELL 62 (code) ---
# ─── Qwen inference helper ────────────────────────────────────────────────────
def qwen_chat(system_prompt: str, user_message: str, max_new_tokens: int = 256) -> str:
    """Send a chat message to Qwen2.5 and return the response text."""
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user',   'content': user_message}
    ]
    text = qwen_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = qwen_tokenizer([text], return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        generated_ids = qwen_model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=qwen_tokenizer.eos_token_id
        )
    # Decode only the newly generated tokens
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    return qwen_tokenizer.decode(output_ids, skip_special_tokens=True).strip()

--- CELL 63 (code) ---
# ─── Translation: French → English ───────────────────────────────────────────
# Identify reviews without English translation, use corrected French as source
needs_translation = df[df['avis_en'].isna() | (df['avis_en'].str.strip() == '')].copy()
print(f'Reviews needing translation: {len(needs_translation):,}')

TRANSLATE_SYSTEM = (
    'You are a professional translator. Translate the French insurance review to English. '
    'Output ONLY the English translation, nothing else.'
)

def translate_review(idx: int) -> tuple:
    """Translate a single review, returning (index, translation)."""
    # Use corrected French if available, fall back to raw avis
    fr_text = df.loc[idx, 'avis_corrected'] if 'avis_corrected' in df.columns else df.loc[idx, 'avis']
    if not isinstance(fr_text, str) or not fr_text.strip():
        return idx, None
    try:
        translation = qwen_chat(TRANSLATE_SYSTEM, fr_text[:500], max_new_tokens=200)
        return idx, translation
    except Exception:
        return idx, None

# Translate sequentially with progress bar
# Note: Qwen inference is not thread-safe, do not parallelise this
print('Translating... (this will take a while for large datasets)')
translations = {}
for idx in tqdm(needs_translation.index, desc='Translating'):
    _, trans = translate_review(idx)
    if trans:
        translations[idx] = trans

# Fill in translated values
for idx, trans in translations.items():
    df.at[idx, 'avis_en'] = trans

n_success = sum(1 for v in translations.values() if v)
n_failed  = len(needs_translation) - n_success
print(f'\nTranslation complete:')
print(f'  ✅ Translated: {n_success:,}')
print(f'  ❌ Failed:     {n_failed:,}')
print(f'  Total avis_en coverage: {df["avis_en"].notna().sum():,} / {len(df):,} ({df["avis_en"].notna().mean()*100:.1f}%)')

# Show samples
print('\n=== Sample Translations ===')
for idx, trans in list(translations.items())[:3]:
    fr = df.loc[idx, 'avis_corrected'] if 'avis_corrected' in df.columns else df.loc[idx, 'avis']
    print(f'FR: {str(fr)[:150]}')
    print(f'EN: {str(trans)[:150]}')
    print('---')

# ─── Save immediately after translation to avoid losing work ─────────────────
df.to_parquet('outputs/reviews_corrected_and_translated.parquet', index=False)
print(f'\n✅ Saved with translations → outputs/reviews_corrected.parquet')

--- CELL 64 (code) ---
df = pd.read_parquet('outputs/reviews_corrected_and_translated.parquet')

--- CELL 65 (code) ---
# ─── Summarisation: per-insurer review summaries (French) ────────────────────
SUMMARISE_SYSTEM = (
    'Tu es un analyste en assurance. À partir d\'un ensemble d\'avis clients pour un assureur, '
    'rédige un résumé concis en 3 phrases couvrant : la satisfaction globale, les points positifs '
    'récurrents, et les principales plaintes. Sois factuel et objectif. '
    'Réponds en français.'
)

top_5_insurers = df['assureur'].value_counts().head(5).index.tolist()
summaries = {}

for insurer in top_5_insurers:
    # Use corrected French reviews as source
    subset = (
        df[df['assureur'] == insurer]['avis_corrected']
        .dropna()
        .head(20)
        .tolist()
    )
    combined = '\n---\n'.join(subset[:10])  # keep context manageable
    prompt = f'Avis clients pour {insurer}:\n{combined[:1500]}'
    summaries[insurer] = qwen_chat(SUMMARISE_SYSTEM, prompt, max_new_tokens=200)
    print(f'[{insurer}]\n{summaries[insurer]}\n')

# Store as DataFrame
df_summaries = pd.DataFrame(
    [(k, v) for k, v in summaries.items()],
    columns=['assureur', 'ai_summary']
)
display(df_summaries)

--- CELL 66 (markdown) ---
This summary technique uses the Qwen2.5-1.5B-Instruct model to generate summaries of the reviews for each insurer. This method is however not very efficient as it requires a lot of computation power and time and due to a limited context window of the model, we can only use around 20 reviews for each insurer at most which leaves out a lot of data for insurers with thousands of reviews.

--- CELL 67 (code) ---
# ─── Optimal Per-Insurer Summary Pipeline ────────────────────────────────────
# Pipeline:
# 1. Compute rating statistics for the insurer
# 2. Extract recurring themes via TF-IDF keyword extraction
# 3. Select representative reviews via weighted stratified sampling
# 4. Feed structured context to Qwen for a readable 3-5 sentence summary

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import numpy as np

# ── 1. Rating statistics ──────────────────────────────────────────────────────
def get_rating_stats(df: pd.DataFrame, insurer: str) -> dict:
    """Compute rating distribution and key statistics for an insurer."""
    insurer_df = df[df['assureur'] == insurer].dropna(subset=['note'])
    total      = len(insurer_df)

    if total == 0:
        return {}

    stats = {
        'total_reviews':  total,
        'average_rating': round(insurer_df['note'].mean(), 2),
        'pct_positive':   round((insurer_df['note'] >= 4).mean() * 100, 1),
        'pct_neutral':    round((insurer_df['note'] == 3).mean() * 100, 1),
        'pct_negative':   round((insurer_df['note'] <= 2).mean() * 100, 1),
        'distribution':   insurer_df['note'].value_counts().sort_index().to_dict()
    }
    return stats


# ── 2. Theme extraction ───────────────────────────────────────────────────────
THEME_KEYWORDS = {
    'Prix / Tarification':      ['prix', 'tarif', 'cher', 'cotisation', 'augmentation',
                                  'coût', 'prime', 'euros', 'compétitif', 'économique'],
    'Service client':           ['service', 'conseiller', 'téléphone', 'accueil', 'réponse',
                                  'joindre', 'rappeler', 'attente', 'injoignable', 'écoute'],
    'Sinistres / Remboursement':['sinistre', 'remboursement', 'rembourser', 'indemnisation',
                                  'expert', 'dossier', 'déclaration', 'réparation', 'prise charge'],
    'Résiliation / Contrat':    ['résiliation', 'résilier', 'contrat', 'annulation',
                                  'conditions', 'clause', 'souscription', 'renouvellement'],
    'Rapidité / Efficacité':    ['rapide', 'efficace', 'délai', 'immédiat', 'simple',
                                  'facile', 'réactif', 'lent', 'retard', 'attendre'],
    'Fidélité / Ancienneté':    ['ans', 'années', 'fidèle', 'depuis', 'longtemps',
                                  'toujours', 'jamais', 'loyalty'],
}

def extract_themes(
    df: pd.DataFrame,
    insurer: str,
    text_col: str = 'avis_corrected_clean'
) -> dict:
    """
    For each theme, compute what proportion of reviews mention it
    and whether it skews positive or negative.
    Returns dict of {theme: {'mention_rate': float, 'avg_rating': float}}
    """
    insurer_df = df[df['assureur'] == insurer].dropna(subset=[text_col, 'note'])
    theme_stats = {}

    for theme, keywords in THEME_KEYWORDS.items():
        # Build regex pattern for any keyword
        pattern = '|'.join(keywords)
        mask    = insurer_df[text_col].str.contains(pattern, case=False, na=False)
        subset  = insurer_df[mask]

        if len(subset) == 0:
            continue

        theme_stats[theme] = {
            'mention_count': len(subset),
            'mention_rate':  round(len(subset) / len(insurer_df) * 100, 1),
            'avg_rating':    round(subset['note'].mean(), 2),
            'skew':          'positif' if subset['note'].mean() >= 3.5 else 'négatif'
        }

    # Sort by mention rate descending
    return dict(sorted(theme_stats.items(), key=lambda x: -x[1]['mention_rate']))


# ── 3. Weighted stratified review selection ───────────────────────────────────
def weighted_stratified_selection(
    df: pd.DataFrame,
    insurer: str,
    total_reviews: int = 15,
    text_col: str = 'avis_corrected'
) -> list:
    """
    Select reviews proportionally to the actual rating distribution,
    then within each rating group select the reviews closest to that
    group's TF-IDF centroid (most representative of that sentiment tier).

    This ensures:
    - The selection mirrors the true sentiment distribution
    - Each selected review is maximally representative of its tier
    - Both short and long reviews are considered equally
    """
    insurer_df = df[df['assureur'] == insurer].dropna(subset=[text_col, 'note'])
    total_insurer = len(insurer_df)

    if total_insurer == 0:
        return []

    # Compute how many reviews to select per rating proportionally
    rating_counts = insurer_df['note'].value_counts().sort_index()
    rating_proportions = (rating_counts / total_insurer * total_reviews).round().astype(int)

    # Ensure at least 1 review per rating that exists, adjust total if needed
    rating_proportions = rating_proportions.clip(lower=1)

    selected_reviews = []

    for note, n_select in rating_proportions.items():
        group_reviews = insurer_df[insurer_df['note'] == note][text_col].tolist()

        if not group_reviews:
            continue

        if len(group_reviews) <= n_select:
            selected_reviews.extend([(note, r) for r in group_reviews])
            continue

        # Select centroid-closest reviews within this rating group
        vectorizer   = TfidfVectorizer(max_features=3000)
        tfidf_matrix = vectorizer.fit_transform(group_reviews)
        centroid     = np.asarray(tfidf_matrix.mean(axis=0))
        similarities = cosine_similarity(tfidf_matrix, centroid).flatten()
        top_indices  = similarities.argsort()[::-1][:n_select]

        selected_reviews.extend([(note, group_reviews[i]) for i in top_indices])

    return selected_reviews  # list of (star_rating, review_text) tuples


# ── 4. Build structured prompt and generate summary ──────────────────────────
OPTIMAL_SYSTEM = (
    'Tu es un analyste en assurance française. '
    'Rédige 3 à 5 phrases en français, '
    'en texte continu sans listes ni titres, '
    'décrivant cet assureur'
    'Base-toi uniquement sur les données fournies. '
    'Utilise uniquement la troisième personne du singulier. '
    'Décris les thèmes en langage naturel sans copier leurs noms. '
    'Si la note est inférieure à 3/5, avertis clairement le lecteur.'
)

def build_structured_prompt(
    insurer: str,
    stats: dict,
    themes: dict,
    selected_reviews: list
) -> str:
    """
    Build a rich structured prompt combining statistics, themes,
    and representative reviews for Qwen to synthesise.
    """
    # Rating stats block
    dist_str = ', '.join([f'{int(k)}★: {v} avis' for k, v in stats['distribution'].items()])
    stats_block = (
        f'=== STATISTIQUES ===\n'
        f'Assureur: {insurer}\n'
        f'Nombre total d\'avis: {stats["total_reviews"]:,}\n'
        f'Note moyenne: {stats["average_rating"]}/5\n'
        f'Répartition: {dist_str}\n'
        f'% Positifs (4-5★): {stats["pct_positive"]}% | '
        f'% Neutres (3★): {stats["pct_neutral"]}% | '
        f'% Négatifs (1-2★): {stats["pct_negative"]}%\n'
    )

    # Themes block
    themes_block = '=== THÈMES RÉCURRENTS ===\n'
    for theme, data in list(themes.items())[:6]:
        themes_block += (
            f'• {theme}: mentionné dans {data["mention_rate"]}% des avis, '
            f'note moyenne {data["avg_rating"]}/5 ({data["skew"]})\n'
        )

    # Representative reviews block, grouped by rating
    reviews_block = '=== AVIS REPRÉSENTATIFS ===\n'
    for note, review in sorted(selected_reviews, key=lambda x: x[0]):
        # Truncate individual reviews to avoid exceeding context window
        truncated = review[:300] + '...' if len(review) > 300 else review
        reviews_block += f'\n[{int(note)}★] {truncated}\n'

    return f'{stats_block}\n{themes_block}\n{reviews_block}'


def optimal_insurer_summary(
    df: pd.DataFrame,
    insurer: str,
    text_col_raw: str = 'avis_corrected',
    text_col_clean: str = 'avis_corrected_clean'
) -> str:
    """
    Full pipeline: stats → themes → weighted stratified selection → Qwen summary.
    """
    # Step 1: Rating statistics
    stats = get_rating_stats(df, insurer)
    if not stats:
        return f'Données insuffisantes pour {insurer}.'

    # Step 2: Theme extraction
    themes = extract_themes(df, insurer, text_col=text_col_clean)

    # Step 3: Weighted stratified review selection
    selected_reviews = weighted_stratified_selection(
        df, insurer, total_reviews=15, text_col=text_col_raw
    )

    # Step 4: Build prompt and generate
    prompt = build_structured_prompt(insurer, stats, themes, selected_reviews)
    return qwen_chat(OPTIMAL_SYSTEM, prompt, max_new_tokens=400)


# ── 5. Run on top 5 insurers ──────────────────────────────────────────────────
top_5_insurers = df['assureur'].value_counts().head(5).index.tolist()

print('Generating optimal summaries for top 5 insurers...\n')
insurer_summaries = {}

for insurer in top_5_insurers:
    print(f'{"="*70}')
    print(f'INSURER: {insurer}')
    print('="*70}')

    # Also print the stats and themes for interpretability
    stats  = get_rating_stats(df, insurer)
    themes = extract_themes(df, insurer)

    print(f'  Average rating: {stats["average_rating"]}/5  |  '
          f'Positive: {stats["pct_positive"]}%  |  '
          f'Negative: {stats["pct_negative"]}%')
    print(f'  Top themes: '
          + ', '.join([f'{t} ({d["skew"]})' for t, d in list(themes.items())[:3]]))
    print()

    summary = optimal_insurer_summary(df, insurer)
    insurer_summaries[insurer] = summary
    print(f'SUMMARY:\n{summary}\n')

# ── 6. Store results ──────────────────────────────────────────────────────────
df_summaries = pd.DataFrame(
    [(k, v) for k, v in insurer_summaries.items()],
    columns=['assureur', 'ai_summary']
)
display(df_summaries)

--- CELL 68 (markdown) ---
## Per-Insurer Summary Generation

### Approach and Methodology

Generating meaningful per-insurer summaries from thousands of reviews presents a fundamental scaling challenge: large language models have finite context windows, making it impossible to feed all available reviews directly. We designed a multi-stage pipeline to address this while maximising the representativeness and accuracy of the final output.

The pipeline consists of four sequential steps:

**Step 1, Rating Statistics.** For each insurer we compute the total review count, average rating, and the proportion of positive (4-5★), neutral (3★), and negative (1-2★) reviews. This quantitative grounding ensures the final summary reflects the true sentiment distribution rather than being biased by whichever reviews happen to be selected.

**Step 2, Theme Extraction.** We identify six recurring themes (pricing, customer service, claims/reimbursement, cancellation/contract, speed/efficiency, and loyalty/tenure) using keyword matching on `avis_corrected_clean`. For each theme we compute the mention rate across all reviews and the average star rating of reviews mentioning it, which tells us not just what customers discuss but whether each topic is a source of satisfaction or complaint for that specific insurer.

**Step 3, Weighted Stratified Review Selection.** Rather than feeding raw reviews to the LLM, we select 15 representative reviews proportionally to the actual rating distribution. Within each star rating group, we use TF-IDF centroid similarity to select the reviews closest to that group's average, the most "typical" reviews for each sentiment tier. This approach uses the full corpus for the selection process while keeping the context sent to the LLM manageable. It also guarantees that the selection mirrors the true positive/negative balance of the insurer's review profile rather than being dominated by long complaint reviews, which TF-IDF centroid methods tend to favour when applied globally.

**Step 4, LLM Synthesis.** The statistics, theme breakdown, and selected reviews are assembled into a structured prompt and fed to Qwen2.5-1.5B-Instruct, which generates a 3–5 sentence paragraph in French describing the insurer to a prospective client.

---

### Difficulties Encountered

The summary generation process required extensive iteration due to the limitations of Qwen2.5-1.5B-Instruct for complex constrained generation. Several failure modes were encountered and addressed:

**Second-person address.** Early prompt versions produced summaries addressing the reader as "vous" ("Dans votre situation, vous êtes satisfait...") rather than describing the insurer in the third person. This is a default behaviour of instruction-tuned models trained on conversational data, requiring an explicit constraint in the system prompt.

**Theme name copying.** When all themes for an insurer were negative (as for Néoliane and GMF), the model defaulted to listing theme taxonomy names verbatim rather than synthesising them in natural language, producing outputs like "les problèmes incluent le service client, la fidélité, les sinistres" which adds no analytical value.

**Hallucinated statistics.** Some iterations produced fabricated percentages or average ratings not present in the input data, particularly for the conclusion sentence. This is a known failure mode of small LLMs when asked to synthesise quantitative information, they interpolate plausible-sounding numbers rather than faithfully reporting the provided figures.

**Factual inversions.** For Direct Assurance, whose top two themes are both negative, several prompt versions produced summaries praising the very aspects customers most complained about. The model was defaulting to generic insurance marketing language from its training distribution rather than grounding in the provided data.

**Model size ceiling.** We attempted to upgrade to Qwen3.5-4B to address these issues, but the model architecture (`qwen3_5`) was not yet supported by the installed version of the transformers library. Given this constraint, we retained Qwen2.5-1.5B-Instruct with a simplified, focused system prompt that proved more reliable than longer prompts with many simultaneous constraints, a well-documented phenomenon for small models where prompt complexity degrades instruction adherence.

---

### Results Analysis

**Direct Assurance (3.31/5, 51.9% positive, 29.3% negative)**
The summary correctly identifies rapid subscription and simplicity as the key positives, and names résiliation difficulties and poor customer service as the main complaints. The conclusion that it offers "un niveau de satisfaction modéré" is accurate. One minor inaccuracy: "critiques positives concernant le rapport qualité-prix" slightly contradicts the theme data showing pricing as a negative theme. Overall a reliable summary for a prospective client.

**L'Olivier Assurance (3.83/5, 71.0% positive, 16.7% negative)**
The summary correctly reflects the predominantly positive profile and identifies the three positive themes (price, service, speed). The hallucination about "gestion efficace des sinistres" is the main weakness, the sinistres theme is not among L'Olivier's positives in the data. Despite this, the overall picture conveyed is broadly accurate for an insurer with 71% positive reviews.

**APRIL Moto (3.95/5, 72.8% positive, 12.3% negative)**
Accurate and well-balanced. The positives (price, speed, service client) match the theme data exactly. The negatives (processus administratifs, délais) are real minority complaint themes expressed in natural language. The "nous pouvons conclure" first-person plural is a minor stylistic inconsistency. This is the most reliable summary of the batch.

**GMF (2.88/5, 38.2% positive, 43.8% negative)**
The strongest output. The summary opens by correctly flagging the problematic profile, names concrete problems in natural language (difficultés lors des sinistres, retards, aspects financiers), acknowledges the minority of positive experiences without overstating them, and concludes with an honest assessment: "malgré quelques points forts, le GMF reste généralement reconnu pour ses inconvénients." This is exactly the kind of frank summary that would genuinely inform a prospective client's decision. At six sentences it slightly exceeds the target length but the additional content is justified by the complexity of the profile.

**Néoliane Santé (2.8/5, 37.4% positive, 43.3% negative)**
The weakest output. At four sentences it is too brief for an insurer with three negative top themes. "La majorité des avis positifs concerne les services de conseil et le rapport qualité-prix" is vague and not strongly supported by the data, Néoliane's positive reviews are weak and scattered without a clear consistent strength. The model has not delivered the explicit warning that a sub-3.0 rating warrants, despite the system prompt instructing it to do so. This is a direct consequence of the model size ceiling: with a larger model, the warning instruction would be reliably followed.

---

### Conclusion

The pipeline successfully produces useful and broadly accurate summaries for four of the five insurers. The remaining imperfections are attributable to the 1.5B parameter scale of the LLM rather than to the pipeline architecture, which is sound. The weighted stratified selection and structured prompt approach represent a principled solution to the scaling problem and are fully reproducible. For production use, replacing Qwen2.5-1.5B with a 4B+ model would resolve the remaining hallucination and instruction-adherence issues identified above.

--- CELL 69 (code) ---
# ─── Export clean master file ─────────────────────────────────────────────────
os.makedirs('outputs', exist_ok=True)

# Build clean DataFrame with all processed columns
cols_to_keep = [
    'note', 'auteur', 'avis', 'assureur', 'produit', 'type',
    'date_publication', 'date_exp',
    'avis_en',               # original / newly translated English
    'avis_clean',            # cleaned French (no stops, no punctuation)
    'avis_en_clean',         # cleaned English
    'avis_light',            # light-clean French (keeps stopwords, apostrophes preserved)
    'avis_corrected',        # spelling-corrected raw French
    'avis_corrected_clean',  # spelling-corrected + normalised (for TF-IDF, Word2Vec)
    'avis_corrected_light',  # spelling-corrected + light normalised (for BERT, LSTM)
    'sentiment',             # derived sentiment label (positive/neutral/negative)
]
cols_present = [c for c in cols_to_keep if c in df.columns]
df_clean = df[cols_present].copy()

# Derive sentiment label
def rating_to_sentiment(note):
    if note >= 4:   return 'positive'
    elif note == 3: return 'neutral'
    else:           return 'negative'

df_clean['sentiment'] = df_clean['note'].apply(rating_to_sentiment)

# Save
df_clean.to_csv('outputs/reviews_clean.csv', index=False, encoding='utf-8-sig')
print(f'Clean file saved → outputs/reviews_clean.csv  ({len(df_clean):,} rows, {df_clean.shape[1]} columns)')
display(df_clean.head(3))

--- CELL 70 (markdown) ---
<a id='5'></a>
## 5. Section 3, Topic Modeling (2 points)

We use **Latent Dirichlet Allocation (LDA)** via Gensim to discover latent topics in the review corpus. We:
1. Build a Gensim dictionary and bag-of-words corpus
2. Train an LDA model, sweeping over number of topics to select best coherence score
3. Visualise topics with `pyLDAvis` and word clouds per topic
4. Assign dominant topic to each review

--- CELL 71 (code) ---
df = pd.read_csv('outputs/reviews_clean.csv')

--- CELL 72 (code) ---
# ─── Prepare corpus for LDA ───────────────────────────────────────────────────
lda_texts = [text.split() for text in df['avis_corrected_clean'].dropna() if len(text.split()) >= 5]

# Build dictionary, filter extremes
dictionary = Dictionary(lda_texts)
dictionary.filter_extremes(no_below=5, no_above=0.6)
corpus = [dictionary.doc2bow(text) for text in lda_texts]

print(f'Vocab size: {len(dictionary):,}')
print(f'Corpus size: {len(corpus):,}')

--- CELL 73 (code) ---
# ─── Coherence sweep to find optimal number of topics ────────────────────────
coherence_scores = {}
for n_topics in [4, 5, 6, 7, 8, 10]:
    lda_tmp = LdaModel(
        corpus, num_topics=n_topics, id2word=dictionary,
        passes=10, random_state=42, alpha='auto', eta='auto'
    )
    cm_model = CoherenceModel(model=lda_tmp, texts=lda_texts, dictionary=dictionary, coherence='c_v')
    coherence_scores[n_topics] = cm_model.get_coherence()
    print(f'  n_topics={n_topics:2d}  coherence={coherence_scores[n_topics]:.4f}')

best_n = max(coherence_scores, key=coherence_scores.get)
print(f'\n✅ Best number of topics: {best_n} (coherence={coherence_scores[best_n]:.4f})')

# Plot coherence
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(list(coherence_scores.keys()), list(coherence_scores.values()), 'o-', color='steelblue')
ax.axvline(best_n, linestyle='--', color='red', label=f'Best n={best_n}')
ax.set_xlabel('Number of Topics')
ax.set_ylabel('Coherence Score (c_v)')
ax.set_title('LDA Coherence vs. Number of Topics')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/lda_coherence.png', bbox_inches='tight')
plt.show()

--- CELL 74 (markdown) ---
Overall Coherence Range
All scores fall in a tight band between 0.445 and 0.476, which is a narrow range indicating the number of topics does not dramatically affect model quality across this range. This is typical for insurance review corpora where the underlying thematic structure is relatively constrained, customers discuss a limited set of topics (price, service, claims, speed) regardless of how many topics you ask the model to find.

The Curve Shape
The curve is non-monotonic with two local peaks at n=5 (0.471) and n=8 (0.476), separated by a local minimum at n=7 (0.459). This double-peak pattern suggests the corpus has two natural levels of thematic granularity:

At 5 topics the model finds broad thematic clusters that align well with the major complaint/satisfaction categories we identified in the EDA, pricing, service, claims, speed, and overall satisfaction
At 8 topics the model achieves the highest coherence by splitting some of these broad categories into more granular sub-topics, for example "service client" might split into "telephone responsiveness" and "claims advisor quality", or "pricing" might split into "initial pricing" and "price increases over time"


Best n=8 Selection
The algorithm correctly selects n=8 as the global maximum at 0.476. The improvement over n=5 is modest, only 0.005 coherence points, which means both are defensible choices. If interpretability is the priority, n=5 produces cleaner, more distinct topics that are easier to label and explain in a presentation. If granularity is the priority, n=8 captures more nuanced distinctions in the corpus.

The Sharp Drop at n=10
The drop to 0.445 at n=10 is the most informative signal in the chart. It indicates that forcing 10 topics causes the model to split coherent thematic clusters into incoherent fragments, there simply are not 10 meaningfully distinct topics in this corpus. This confirms that the true number of underlying themes is somewhere between 5 and 9, and that the algorithm correctly stopped the sweep before this degradation zone.

Practical implication
The algorithm will train the final model with n=8. This is the statistically justified choice. However when you inspect the 8 topic word clouds and topic labels in the next cell, verify that all 8 topics are genuinely interpretable and distinct. If two topics appear to overlap significantly, sharing many top words, consider manually overriding to n=5 for a cleaner presentation, since the coherence difference is negligible and 5 topics are much easier to explain to an audience in a 5-minute video.

--- CELL 75 (code) ---
# ─── Train final LDA model ────────────────────────────────────────────────────
N_TOPICS = best_n

lda_model = LdaModel(
    corpus, num_topics=N_TOPICS, id2word=dictionary,
    passes=20, random_state=42, alpha='auto', eta='auto'
)

# Print topics
print('=== Discovered Topics ===')
topic_labels = {}
for i, topic in lda_model.print_topics(num_words=10):
    print(f'Topic {i}: {topic}')
    # Assign manual label based on dominant words
    top_words_t = [w.split('"')[1] for w in topic.split('+')[:3]]
    topic_labels[i] = ' / '.join(top_words_t)

--- CELL 76 (markdown) ---
These 8 topics are highly interpretable and map cleanly onto real insurance customer experience categories. Here is the full interpretation:

Topic 0, Auto Claims & Incidents
Dominated by "assurance", "sinistre", "véhicule", "voiture", "non" (as in non-responsible). This is the auto insurance claims topic, customers describing accidents, damage, and claim declarations. The presence of "ans" and "chez" suggests the temporal loyalty framing we identified earlier ("client depuis X ans chez cet assureur"). This topic captures the experience of actually using auto insurance rather than just subscribing to it.

Topic 1, Pricing Complaints & Price Increases
"Plus", "moins", "cher", "année", "contrat", "ans", "autres", this is the price dissatisfaction topic, specifically about price increases over time. "Plus cher" (more expensive), "moins cher" (less expensive/competitors), "année" and "ans" confirm this is about premiums increasing year over year. The presence of "contrats" (plural) suggests customers comparing multiple contracts or noting price differences across their portfolio. This maps directly to the "betrayed loyalty" pattern, long-term customers who feel penalised financially for staying.

Topic 2, General Positive Satisfaction
The clearest and most coherent topic: "très", "prix", "assurance", "service", "satisfait", "rapide", "bien", "recommande", "merci", "simple". This is the formulaic positive review topic we identified extensively in the n-gram analysis, "très satisfait, service rapide, prix compétitif, je recommande". The very high weight on "très" (0.070) confirms our earlier finding that degree adverbs are the primary marker of positive reviews. This topic is essentially the positive sentiment class made visible.

Topic 3, Customer Service Failures
"Téléphone", "service", "client", "faire", "jamais", "personne", "tout", this is the negative customer service topic. "Jamais" (never), "personne" (nobody/no one) and "quand" in a complaints context typically appears as "quand j'appelle, personne ne répond" or "jamais de réponse". The contrast with Topic 7 (positive phone contact) is striking and confirms that telephone customer service is polarising, it is both a top source of praise and a top source of complaint in this corpus.

Topic 4, Health Insurance & Mutual Reimbursements
The most specific and coherent topic: "mutuelle", "remboursements", "santé", "remboursement", "mgen", "soins", "adhérent", "charge", "mgp". This is exclusively about health insurance (mutuelle santé) reimbursements. The presence of specific insurer names MGEN and MGP confirms this is a product-specific topic rather than a general theme. This topic would not have been visible without the n=8 granularity, at n=5 it would likely have merged with general claims/reimbursement content.

Topic 5, Administrative Delays & Unresolved Requests
"Mois", "contrat", "demande", "après", "toujours", "fait", "depuis", "jours", this is the temporal grievance topic we identified extensively in the EDA. "Depuis X mois, toujours rien" is the canonical negative review structure captured here. "Demande" (request/claim), "après" (after), "compte" (account/expectation) confirm this is about unresolved administrative situations. The distinction between this topic and Topic 3 is subtle but meaningful, Topic 3 is about the quality of human interaction (nobody answers the phone), while Topic 5 is about process failures over time (months of waiting with no resolution).

Topic 6, Life Insurance, Disability & Provident Products
"Vie", "dossier", "arrêt", "travail", "décès", "maladie", "prévoyance", "cardif", "invalidité", a highly specific topic covering life insurance, disability claims, and provident products (prévoyance). The presence of "cardif" (BNP Paribas Cardif, a major French life insurer) as a named entity is notable. This is the most serious complaint topic in the corpus, customers dealing with disability, work stoppage, and death benefits, which align with the very long, emotionally charged negative reviews we saw in the extractive summary results for Allianz.

Topic 7, Positive Telephone & Human Contact
"Très", "écoute", "téléphonique", "accueil", "téléphone", "contact", "merci", "agréable", "questions", "bon", the positive counterpart of Topic 3. This captures customers praising the quality of human interaction: attentive listening ("écoute"), warm reception ("accueil agréable"), helpful responses to questions. The very high weight on "très" (0.088, the highest of any topic) and "écoute" together confirm that being listened to is the single most valued quality in positive telephone experiences.

Overall Assessment
The 8-topic model has produced an excellent and interpretable thematic decomposition of the corpus. The topics are distinct, semantically coherent, and directly mappable to real insurance customer experience categories. Crucially the model has discovered product-specific topics (Topic 4 for health, Topic 6 for life/prévoyance) alongside cross-product themes (Topics 2 and 3), which validates the n=8 choice over n=5, the additional granularity is genuinely informative rather than arbitrary splitting.

The one remaining issue is the stopword leakage we flagged, "très", "plus", "bien", "tout", "fait", "chez", "avoir", "faire", "après", "depuis", "jamais", "quand" all appear as top words in multiple topics. Their presence dilutes the discriminative power of Topics 0, 1, 3, and 5 in particular. Topics 4, 6, and 7 are the cleanest because their domain-specific vocabulary (mutuelle, prévoyance, écoute) naturally dominates over the leaked stopwords. This is the most important thing to mention as a limitation in our results interpretation section.

--- CELL 77 (code) ---
pyLDAvis.enable_notebook()
vis = gensimvis.prepare(lda_model, corpus, dictionary, sort_topics=False)
pyLDAvis.save_html(vis, 'outputs/lda_vis.html')
print('pyLDAvis saved to outputs/lda_vis.html')

# ─── Print top words per topic ────────────────────────────────────────────────
print('\n=== Top Words per Topic ===\n')
for topic_id in range(N_TOPICS):
    top_words = lda_model.show_topic(topic_id, topn=10)
    words_str = ', '.join([f'{word} ({prob:.3f})' for word, prob in top_words])
    label     = topic_labels.get(topic_id, f'Topic {topic_id}')
    print(f'Topic {topic_id} [{label}]')
    print(f'  {words_str}')
    print()

--- CELL 78 (markdown) ---
## LDA Topic Modeling Results

### Discovered Topics

The final LDA model with 8 topics, selected via coherence sweep as the optimal 
number, reveals a coherent and interpretable thematic decomposition of the 
insurance review corpus. Each topic maps cleanly onto a recognisable customer 
experience category:

**Topic 0, Sinistres Auto** `[assurance / sinistre / véhicule / voiture]`
Clearly centred on auto insurance claim experiences. "Non" appears in "non 
responsable" constructions, confirming this topic captures the experience of 
being involved in accidents the customer was not at fault for. Stopword leakage 
is visible in "plus", "tout", "fait", "chez" but the domain signal is strong 
enough to identify the topic unambiguously.

**Topic 1, Augmentation des Prix** `[plus cher / année / contrat / ans]`
The pricing complaint topic, specifically about premiums increasing over time. 
"Plus cher", "moins cher", "année après année", "depuis X ans" are the canonical 
constructions captured here. The dual presence of "contrat" and "contrats" 
(singular and plural) suggests customers comparing their current contract to 
competitors or to past pricing, the betrayed loyalty pattern we identified 
extensively in the n-gram analysis.

**Topic 2, Satisfaction Générale** `[très satisfait / prix / service / rapide / recommande]`
The formulaic positive review topic and the direct empirical confirmation of our 
earlier EDA finding that positive reviews are lexically concentrated and 
repetitive. Every content word here is a positive signal. "Très" at 0.070 and 
"satisfait" at 0.039 together make this essentially a "very satisfied" topic, 
the benchmark positive class for the supervised learning section.

**Topic 3, Service Client Négatif** `[téléphone / service client / jamais / personne]`
The negative telephone experience topic. "Jamais" (0.013) and "personne" (0.013) 
at equal weight confirms the canonical construction "jamais personne au téléphone" 
— nobody ever answers. "Quand" at 0.014 appears in "quand j'appelle" 
constructions. The contrast with Topic 7 is the most analytically interesting 
finding from the entire model, telephone customer service is simultaneously the 
top source of praise and the top source of complaint in this corpus.

**Topic 4, Mutuelle Santé** `[mutuelle / remboursements / santé / mgen / mgp]`
The cleanest and most specific topic in the model. "Mutuelle" at 0.126 is by far 
the highest single-word probability across all 8 topics, this topic is 
unmistakably about health insurance mutuals. MGEN and MGP appearing as named 
entities confirms the model has correctly isolated the health insurance segment. 
"Adhérent" (member) rather than "client" is characteristic of mutuelles which 
use membership rather than commercial language, a subtle but genuine linguistic 
distinction the model has captured.

**Topic 5, Délais Administratifs** `[mois / contrat / demande / toujours / depuis / jours]`
The temporal grievance topic. "Mois" (0.031), "depuis" (0.013), "toujours" 
(0.014), "jours" (0.009) together construct the "depuis X mois, toujours rien" 
pattern identified in the n-gram analysis as the canonical negative review 
structure. "Compte" likely appears in "prise en compte", customers waiting for 
their request to be acknowledged by the insurer.

**Topic 6, Prévoyance & Vie** `[vie / arrêt travail / décès / maladie / prévoyance / cardif / invalidité]`
The most emotionally serious topic in the corpus, covering life insurance, 
disability claims, and provident products. Every content word represents a 
significant life event, work stoppage, death, illness, disability. The 
relatively low probability weights (all below 0.020) reflect that this is a 
niche topic with specialised vocabulary that is individually rare but 
collectively coherent. Cardif appearing as a named entity confirms this maps 
primarily to BNP Paribas Cardif customers dealing with life and disability 
products, consistent with the very long, emotionally charged negative reviews 
surfaced by the extractive summarisation in Section 2.

**Topic 7, Accueil Téléphonique Positif** `[très / écoute / téléphonique / accueil / agréable / merci]`
The direct counterpart to Topic 3 and one of the most analytically significant 
findings of the model. "Très" at 0.088, the highest weight of any word in the 
entire model, combined with "écoute" (attentiveness) at 0.036 confirms our EDA 
finding that being listened to is the single most valued quality in positive 
insurance interactions. "Agréable" and "accueil" together paint a consistent 
picture of warm, attentive service. The existence of Topics 3 and 7 as 
distinct poles confirms that telephone customer service is the most 
sentiment-polarising dimension of the insurance customer experience.

---

### pyLDAvis Intertopic Distance Map

The pyLDAvis visualisation reveals the thematic structure of the corpus 
spatially. Topics are positioned by multidimensional scaling of their word 
distributions, the further apart two topics are, the more distinct their 
vocabularies.

**Topic overlap in the central cluster.** The most striking feature of the map 
is the cluster of overlapping topics 1, 2, 4, and 6 at the centre. These four 
topics share significant vocabulary, "assurance", "très", "service", "contrat", 
"mois" appear across all of them, pulling their centroids together in the MDS 
projection. This overlap is the visual confirmation of the stopword leakage 
problem identified in the cleaning section: quasi-stopwords that survived the 
NLTK French stopword list ("très", "plus", "bien", "toujours", "cette", "chez") 
contribute to topic similarity rather than topic discrimination. An extended 
custom French stopword list would resolve this overlap in future iterations.

**Well-separated niche topics.** Topics 3, 5, 7, and 8 (using pyLDAvis 
numbering) are well-separated from the central cluster and from each other, 
indicating genuinely distinct vocabularies. These correspond to our most 
interpretable topics, customer service failures, life insurance/disability, 
administrative delays, and positive telephone contact, whose isolation in the 
map confirms that their domain-specific vocabulary (mutuelle, prévoyance, 
écoute, téléphonique) naturally dominates over the leaked stopwords.

**Topic sizes.** The bubble sizes reflect the marginal topic distribution. 
Topics 1, 2, and 6 are the largest, capturing the two most common review 
archetypes identified in the EDA: the formulaic positive review and the general 
auto insurance claims experience. The smaller Topic 4 (Mutuelle Santé) and 
Topic 7 (Positive Telephone Contact) are consistent with their niche scope.

**Most salient terms.** The right panel of the pyLDAvis visualisation confirms 
the stopword leakage conclusively, "très", "plus", "bien", "toujours", "cette", 
"chez" all appear in the top 30 most salient terms globally. The genuinely 
discriminative terms are "mutuelle", "sinistre", "remboursement", "écoute", 
"téléphonique", "dossier", the words that actually separate topics from each 
other. "Assurance" and "contrat" are the two most frequent terms overall, 
explaining their presence across multiple overlapping topics.

---

### Overall Assessment

The 8-topic model reveals a two-tier thematic structure in the corpus: four 
well-separated, highly coherent niche topics surrounded by a cluster of 
overlapping general topics sharing common insurance vocabulary. The most 
important analytical finding is the Topics 3/7 polarity, telephone customer 
service is the most sentiment-polarising dimension of the insurance experience, 
appearing as both the primary source of praise and the primary source of 
complaint. Topic 4 (Mutuelle Santé) and Topic 6 (Prévoyance & Vie) are 
particularly valuable as they identify product-specific thematic clusters that 
would have been invisible at n=5, validating the coherence-based model selection 
approach. The stopword leakage, while creating visual overlap in the distance 
map, did not prevent the model from discovering the most meaningful thematic 
distinctions in the corpus.

--- CELL 79 (code) ---
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
# ─── Word cloud per topic ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, (N_TOPICS + 1) // 2, figsize=(16, 8))
axes = axes.flatten()

for topic_id in range(N_TOPICS):
    word_weights = dict(lda_model.show_topic(topic_id, topn=50))
    wc = WordCloud(width=300, height=200, background_color='white',
                   colormap='viridis').generate_from_frequencies(word_weights)
    axes[topic_id].imshow(wc, interpolation='bilinear')
    axes[topic_id].axis('off')
    axes[topic_id].set_title(f'Topic {topic_id}: {topic_labels.get(topic_id, "")}', fontsize=9)

# Hide unused subplots
for i in range(N_TOPICS, len(axes)):
    axes[i].axis('off')

plt.suptitle('LDA Topic Word Clouds', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/lda_topic_wordclouds.png', bbox_inches='tight')
plt.show()

--- CELL 80 (code) ---
# ─── Assign dominant topic to each review ─────────────────────────────────────
def get_dominant_topic(bow):
    topic_probs = lda_model.get_document_topics(bow, minimum_probability=0)
    return max(topic_probs, key=lambda x: x[1])[0]

# We only have corpus aligned to the filtered lda_texts, re-align
df_lda = df.loc[df['avis_corrected_clean'].dropna().index[:len(corpus)]].copy()
df_lda['dominant_topic'] = [get_dominant_topic(bow) for bow in corpus]
df_lda['topic_label'] = df_lda['dominant_topic'].map(topic_labels)

# Merge back
df = df.merge(df_lda[['dominant_topic', 'topic_label']], left_index=True, right_index=True, how='left')

# Topic distribution
fig, ax = plt.subplots(figsize=(10, 4))
df['dominant_topic'].value_counts().sort_index().plot.bar(ax=ax, color='mediumslateblue')
ax.set_title('Review Count per Dominant LDA Topic')
ax.set_xlabel('Topic ID')
ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig('outputs/lda_topic_distribution.png', bbox_inches='tight')
plt.show()

display(df[['avis_corrected', 'dominant_topic', 'topic_label']].head(6))

--- CELL 81 (markdown) ---
### Topic Distribution Chart
The distribution confirms and quantifies what the pyLDAvis bubble sizes suggested. Three topics dominate the corpus:

Topic 2: Satisfaction Générale – The largest with approximately 11,000 reviews, making it the single most common review archetype in the dataset. This is consistent with our EDA finding that positive reviews are formulaic and concentrated around a small set of repeated phrases; their lexical uniformity makes them easy for LDA to cluster together into one dominant topic.
Topic 0: Sinistres Auto – The second largest at approximately 8,900 reviews, reflecting the 58.6% auto insurance product dominance we identified in the EDA. Auto insurance customers who experience claims generate a high volume of detailed reviews, and this topic captures nearly a third of the entire corpus.
Topic 5: Délais Administratifs – At approximately 4,900 reviews, this is the third largest, confirming that unresolved administrative situations are a pervasive complaint pattern across all insurers and products rather than being specific to any one insurer.
Topic 3: Service Client Négatif (~3,600 reviews) and Topic 1: Augmentation des Prix (~3,200 reviews) – Mid-sized, confirming these are real recurring complaint categories but less dominant than claims and administrative delays.
Topics 4, 6, and 7: Mutuelle Santé, Prévoyance & Vie, and Accueil Téléphonique Positif – The smallest at under 500, 350, and 1,900 reviews respectively.
The very small size of Topics 4 and 6 is expected given their product-specific nature; they capture niche segments of the corpus.
Topic 7’s relatively modest size (1,900 reviews) despite being a positive topic is noteworthy: it confirms that genuinely positive experiences specifically praising telephone contact are less common than generic satisfaction (Topic 2), which makes intuitive sense.
### Per-Review Assignment Validation
The six sample rows at the bottom provide a sanity check on the assignment quality:

Row 0: "La personne au téléphone était Claire et sympathique" → Topic 2 (Satisfaction Générale). This is slightly surprising; it could equally belong to Topic 7 (Accueil Téléphonique Positif) given the telephone context. However, the review is short and generic enough that Topic 2's broader positive vocabulary "wins." This is a legitimate LDA ambiguity for short reviews.
Row 1: "Satisfait. Réactivité, simplicité. Prix attractif" → Topic 2 (Satisfaction Générale). Correct; this is the canonical formulaic positive review.
Row 2: "Assureur à fuir, n'assure pas ses responsabilités" → Topic 0 (Sinistres Auto). Correct; this is a complaint about an insurer failing to cover its responsibilities, consistent with the claims experience topic.
Row 3: "Voilà 3 mois que la GMF me fait attendre" → Topic 0 (Sinistres Auto). This is debatable; the temporal waiting pattern ("3 mois") strongly suggests Topic 5 (Délais Administratifs) would be more appropriate. The assignment to Topic 0 likely occurred because "GMF" and other vocabulary pulled it toward the general insurance claims topic. This is an example where LDA's soft assignment probabilities would be more informative than hard dominant topics; this review is likely split between Topics 0 and 5.
Rows 4 and 5: Both correctly assigned to Topic 2 as generic positive reviews.
### Overall
The topic assignment is working correctly for clear-cut cases and shows expected ambiguity for short or mixed reviews. The distribution is coherent with both the EDA findings and the pyLDAvis bubble sizes, validating the full pipeline.

The dominant topic column added to the dataframe is now a useful feature for the supervised learning section:

It can serve as an additional input feature alongside the text.
The per-topic sentiment analysis in the Results Interpretation section can use it to quantify which topics are most associated with negative versus positive ratings.

--- CELL 82 (markdown) ---
<a id='6'></a>
## 6. Section 4, Embeddings & Similarity (2+ points)

Word embeddings capture semantic meaning in vector space. We train and evaluate:
- **Word2Vec** (Skip-gram, CBOW) trained on our corpus
- **GloVe** pre-trained vectors loaded and adapted
- **Matplotlib** 2D t-SNE and PCA visualisation
- **TensorBoard** embedding projector export
- **Euclidean & Cosine distance** implementation from scratch
- **Semantic search** with FAISS (bonus)

--- CELL 83 (markdown) ---
### 6.1 Word2Vec Training

--- CELL 84 (code) ---
# ─── Prepare sentences for Word2Vec ──────────────────────────────────────────
w2v_sentences = [text.split() for text in df['avis_corrected_clean'].dropna() if len(text.split()) >= 3]
print(f'Training corpus: {len(w2v_sentences):,} sentences')

# Train Skip-gram model
w2v_sg = Word2Vec(
    sentences=w2v_sentences,
    vector_size=100,
    window=5,
    min_count=5,
    workers=4,
    sg=1,         # 1 = skip-gram, 0 = CBOW
    epochs=15,
    seed=42
)

# Train CBOW model
w2v_cbow = Word2Vec(
    sentences=w2v_sentences,
    vector_size=100,
    window=5,
    min_count=5,
    workers=4,
    sg=0,
    epochs=15,
    seed=42
)

print(f'Word2Vec vocabulary size: {len(w2v_sg.wv):,}')
print(f'Most similar to "assurance": {w2v_sg.wv.most_similar("assurance", topn=5)}')

--- CELL 85 (code) ---
# ─── Save and explore Word2Vec ─────────────────────────────────────────────────
w2v_sg.save('outputs/word2vec_skipgram.model')
w2v_cbow.save('outputs/word2vec_cbow.model')

# Analogy test (if vocabulary allows)
test_pairs = [
    ('assurance', 'contrat'),
    ('remboursement', 'sinistre'),
    ('client', 'service'),
]
print('=== Word Similarity (Skip-gram) ===')
for w1, w2 in test_pairs:
    if w1 in w2v_sg.wv and w2 in w2v_sg.wv:
        sim = w2v_sg.wv.similarity(w1, w2)
        print(f'  sim({w1}, {w2}) = {sim:.4f}')

# Top similar words for key insurance terms
key_terms = ['prix', 'sinistre', 'remboursement', 'service', 'contrat']
print('\n=== Most Similar Words ===')
for term in key_terms:
    if term in w2v_sg.wv:
        similar = w2v_sg.wv.most_similar(term, topn=5)
        print(f'  {term:20s}: {similar}')

--- CELL 86 (markdown) ---
### Overall Assessment
The Word2Vec model has learned meaningful semantic relationships for some terms but shows clear quality issues for others. Here is a detailed breakdown:

#### Vocabulary Size
9,800 words from 34,249 sentences is a reasonable vocabulary for this corpus size. The min_count=5 filter has correctly removed rare terms while keeping the insurance-specific vocabulary intact.

#### Most Similar to "assurance"
This result is problematic. "Projets", "sis", "relationnelle", "compagnie" are not natural semantic neighbours of "assurance" in an insurance context. "Assurances" (plural) and "compagnie" are legitimate, but "sis" and "relationnelle" appearing this high suggests "assurance" is too frequent and generic to produce meaningful embeddings, it co-occurs with everything, making its vector the average of the entire corpus rather than a semantically specific representation. This confirms our earlier suggestion to test more discriminative terms instead.

#### Word Similarity Pairs

sim(assurance, contrat) = 0.4434, moderate similarity, plausible since both are general insurance terms that co-occur frequently
sim(remboursement, sinistre) = 0.2172, surprisingly low. These two terms are strongly related in insurance contexts (sinistres lead to remboursements) and should score higher. This suggests they appear in different syntactic contexts, sinistre in complaint narratives and remboursement as an outcome term, meaning they co-occur less directly than expected
sim(client, service) = 0.5541, the most plausible result, "service client" is a fixed bigram that appears thousands of times so these tokens are strongly associated


#### Most Similar Words, The Good

prix → tarif (0.828), tarifs (0.811), excellent. Prix and tarif are genuine synonyms in French insurance vocabulary and the model correctly identifies them as near-identical in meaning. This is the strongest result in the output.
sinistre → sinistres (0.727), accident (0.699), dégât (0.637), very good. The model correctly clusters claim-related vocabulary. "Accident" and "dégât" (damage) are semantically appropriate neighbours for sinistre.
contrat → contrats (0.687), tacite (0.590), good. "Tacite" almost certainly appears in "reconduction tacite" (automatic renewal) which is a frequent complaint topic, making it a contextually appropriate neighbour for contrat.


#### Most Similar Words, The Problematic

remboursement → lentilles (0.654), thermale (0.600), these are health insurance reimbursement types (contact lenses, thermal spa treatment) rather than semantically similar words to remboursement itself. The model has learned that "remboursement" appears frequently in health insurance contexts alongside specific claim types, which is correct co-occurrence learning but not ideal semantic similarity. This is a consequence of the corpus being dominated by auto insurance (58.6%), health insurance reimbursement vocabulary is contextually specific and isolated.
service → élocution (0.659), benjamin (0.628), sévice (0.638), these are clearly noise. "Élocution" (elocution/speech), "benjamin" (a proper name), and "sévice" (abuse, likely a misspelling of "service") should not be neighbours of "service". This indicates the vocabulary contains rare words that co-occurred with "service" in unusual contexts, pulling the embedding in unexpected directions. "Sévice" being close to "service" is a direct spelling correction failure, it is a misspelling that survived into avis_corrected_clean.
prix → couteuse (0.721), expérimenté (0.704), attifais (0.694), "couteuse" (costly) is legitimate, but "expérimenté" and "attifais" are not natural price neighbours. "Attifais" appears to be a rare informal word that co-occurred with price discussions in a small number of reviews.


#### Root Causes
Three factors explain the noise in the results:
Corpus size, 34,249 sentences is on the smaller end for Word2Vec training. Standard recommendations suggest at least 100,000 sentences for reliable embeddings. The 9,800 vocabulary words means some terms have very few training examples, making their vectors unstable.
Residual spelling errors, "sévice", "couteuse", "attifais", "contât", "contreat" appearing in the vocabulary confirm that spelling correction was not comprehensive. These malformed tokens create spurious neighbours for correctly spelled words.
Domain vocabulary isolation, health insurance terms (lentilles, thermale) and life insurance terms are contextually isolated from auto insurance vocabulary, causing the model to learn product-specific co-occurrences rather than general semantic similarity for terms like "remboursement".

#### Conclusion
The model successfully learns morphological relationships (prix/tarif/tarifs, sinistre/sinistres/accident, contrat/contrats/tacite) and some semantic relationships (service/client). The failure cases are informative diagnostics, they reveal residual spelling errors in the corpus, the health/auto insurance vocabulary split, and the generic nature of high-frequency terms like "assurance" and "service". A larger corpus or more aggressive spelling correction would address most of these issues.

--- CELL 87 (markdown) ---
### 6.2 GloVe Pre-trained Vectors

GloVe vectors are loaded from the pre-trained `glove-wiki-gigaword-100` model via Gensim's downloader. We then compare vocabulary coverage with our Word2Vec model.

--- CELL 88 (code) ---
# ─── Load GloVe via Gensim ────────────────────────────────────────────────────
import gensim.downloader as api

print('Downloading GloVe 100d (wiki-gigaword-100)...')
glove_model = api.load('glove-wiki-gigaword-100')
print(f'✅ GloVe loaded  |  Vocab size: {len(glove_model):,}')

# GloVe is English-oriented; test with English review tokens
glove_test_words = ['insurance', 'claim', 'premium', 'coverage', 'customer', 'service']
for w in glove_test_words:
    if w in glove_model:
        similar = glove_model.most_similar(w, topn=5)
        print(f'  GloVe similar to "{w}": {[s[0] for s in similar]}')

--- CELL 89 (code) ---
# ─── GloVe coverage on our corpus ────────────────────────────────────────────
en_tokens = set(' '.join(df['avis_en_clean'].dropna()).split())
in_glove = {t for t in en_tokens if t in glove_model}
coverage = len(in_glove) / max(len(en_tokens), 1) * 100
print(f'GloVe coverage on English review tokens: {coverage:.1f}% ({len(in_glove):,}/{len(en_tokens):,})')

--- CELL 90 (code) ---
# Compare equivalent terms across Word2Vec (French) and GloVe (English)
print('\n=== Word2Vec (French) vs GloVe (English) comparison ===')
comparisons = [('prix', 'price'), ('sinistre', 'claim'), ('service', 'service')]
for fr_word, en_word in comparisons:
    if fr_word in w2v_sg.wv:
        w2v_similar = [w for w, _ in w2v_sg.wv.most_similar(fr_word, topn=3)]
        print(f'  W2V  "{fr_word}": {w2v_similar}')
    if en_word in glove_model:
        glove_similar = [w for w, _ in glove_model.most_similar(en_word, topn=3)]
        print(f'  GloVe "{en_word}": {glove_similar}')
    print()

--- CELL 91 (markdown) ---
### GloVe Coverage, 76.2%
76.2% coverage (16,848 out of 22,098 unique tokens) is a solid result for a translated corpus applied to general-purpose English GloVe vectors. It means roughly 3 in 4 tokens in the English reviews have a pre-trained vector available, which is sufficient for meaningful embedding initialisation in the GloVe+BiLSTM supervised model. The 23.8% miss rate is explained by three factors: insurance-specific jargon without English equivalents ("mutuelle", "sinistre" that survived translation untranslated), informal spelling variants in the translated text, and rare proper nouns (insurer names, place names).

### GloVe Similarity Results
The GloVe results are semantically coherent but reflect a general-purpose English training corpus rather than an insurance-specific one:

insurance → insurers, pension, premiums, care, insurer, correct and domain-relevant
claim → claims, claiming, claimed, deny, denied, excellent, captures both the noun and the dispute context
premium → priced, discount, fare, price, plus, partially correct. "Priced" and "price" are relevant but "fare" (transport pricing) reveals the general corpus bias
coverage → insurance, care, media, benefit, news, problematic. "Media coverage" and "news coverage" are pulling the vector away from insurance coverage, a classic polysemy issue in general-purpose embeddings
customer → customers, employee, user, business, phone, generic but acceptable
service → services, public, network, private, system, clearly dominated by "public service" and "network service" meanings rather than customer service, another polysemy issue


### Word2Vec vs GloVe Comparison, The Key Finding
This is the most analytically valuable output of the two cells:
Prix vs Price, Word2Vec correctly identifies "tarif/tarifs" (exact French synonyms) and "couteuse" (costly). GloVe finds "prices, value, drop", generic price vocabulary. Word2Vec wins here because it learned from insurance-specific co-occurrences where "prix" and "tarif" are used interchangeably, while GloVe's "price" vector is dominated by general economic contexts.
Sinistre vs Claim, Word2Vec finds "sinistres, initie, accident, dégât", the French insurance claim vocabulary cluster. GloVe finds "claims, claiming, claimed, deny, denied", also excellent, capturing both the insurance claim meaning and the dispute/denial context that is common in negative reviews. This is a rare case where GloVe performs comparably to Word2Vec despite the domain mismatch, because "claim" in English is less polysemous than "coverage" or "service".
Service vs Service, This is where the domain gap is most visible. Word2Vec finds "services, élocution, conseilles", the first is correct, the latter two are noise from the small corpus. GloVe finds "services, public, network, private, system", all wrong for the customer service context. Neither model handles "service" well, confirming our earlier finding that this high-frequency term produces unstable embeddings regardless of the approach.

### Overall Conclusion
The comparison demonstrates a consistent pattern: Word2Vec trained on domain-specific French text outperforms general GloVe on insurance-specific semantic relationships (prix/tarif), while GloVe performs comparably or better on terms that are semantically stable across domains (claim/deny). The 76.2% coverage makes GloVe viable as an embedding initialiser for the English supervised model, but the polysemy issues with "coverage" and "service" mean a domain-adapted GloVe or FastText trained on insurance text would produce meaningfully better results. This is worth noting as a limitation and future improvement direction in the results section.

--- CELL 92 (markdown) ---
### 6.3 Matplotlib Embedding Visualisation (PCA + t-SNE)

--- CELL 93 (code) ---
# ─── t-SNE and PCA visualisation of Word2Vec embeddings ──────────────────────
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Select a meaningful subset of words
VIZ_WORDS = [
    'assurance', 'contrat', 'sinistre', 'remboursement', 'prime',
    'prix', 'tarif', 'cher', 'garantie', 'couverture',
    'service', 'conseiller', 'agent', 'téléphone', 'appel',
    'délai', 'rapide', 'attente', 'réponse', 'lenteur',
    'satisfait', 'excellent', 'bien', 'mauvais', 'recommande',
    'sinistre', 'remboursement', 'accident', 'dommage', 'résiliation',
]
VIZ_WORDS = [w for w in VIZ_WORDS if w in w2v_sg.wv]

vectors = np.array([w2v_sg.wv[w] for w in VIZ_WORDS])

# Colour clusters by theme
theme_map = {
    'Contract/Product': ['assurance', 'contrat', 'garantie', 'couverture', 'prime'],
    'Pricing':          ['prix', 'tarif', 'cher'],
    'Service':          ['service', 'conseiller', 'agent', 'téléphone', 'appel'],
    'Speed/Process':    ['délai', 'rapide', 'attente', 'réponse', 'lenteur'],
    'Sentiment':        ['satisfait', 'excellent', 'bien', 'mauvais', 'recommande'],
    'Claims':           ['sinistre', 'remboursement', 'accident', 'dommage', 'résiliation'],
}
word_theme = {}
for theme, words in theme_map.items():
    for w in words:
        word_theme[w] = theme

themes_present = [word_theme.get(w, 'Other') for w in VIZ_WORDS]
unique_themes = list(set(themes_present))
palette = dict(zip(unique_themes, sns.color_palette('tab10', len(unique_themes))))
colours = [palette[t] for t in themes_present]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# PCA
pca = PCA(n_components=2, random_state=42)
pca_result = pca.fit_transform(vectors)
for i, (word, (x, y), col) in enumerate(zip(VIZ_WORDS, pca_result, colours)):
    axes[0].scatter(x, y, color=col, s=60)
    axes[0].annotate(word, (x, y), fontsize=7, ha='center', va='bottom')
axes[0].set_title('PCA Projection of Word2Vec Embeddings')

# Add legend
from matplotlib.patches import Patch
handles = [Patch(color=palette[t], label=t) for t in unique_themes]
axes[0].legend(handles=handles, fontsize=7, loc='best')

# t-SNE
perplexity = min(10, max(2, len(VIZ_WORDS) // 4))
tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
tsne_result = tsne.fit_transform(vectors)
for i, (word, (x, y), col) in enumerate(zip(VIZ_WORDS, tsne_result, colours)):
    axes[1].scatter(x, y, color=col, s=60)
    axes[1].annotate(word, (x, y), fontsize=7, ha='center', va='bottom')
handles2 = [Patch(color=palette[t], label=t) for t in unique_themes]
axes[1].legend(handles=handles2, fontsize=7, loc='best')
axes[1].set_title('t-SNE Projection of Word2Vec Embeddings')

plt.tight_layout()
plt.savefig('outputs/embeddings_visualization.png', bbox_inches='tight')
plt.show()

--- CELL 94 (markdown) ---
### 6.4 TensorBoard Embedding Projector Export

--- CELL 95 (code) ---
# ─── Export embeddings for TensorBoard Projector ─────────────────────────────
import io

TB_DIR = Path('outputs/tensorboard_embeddings')
TB_DIR.mkdir(parents=True, exist_ok=True)

# Export top N words
TOP_N = 500
top_vocab = w2v_sg.wv.index_to_key[:TOP_N]
embedding_matrix = np.array([w2v_sg.wv[w] for w in top_vocab])

# Save vectors.tsv
np.savetxt(TB_DIR / 'vectors.tsv', embedding_matrix, delimiter='\t')

# Save metadata.tsv (word labels)
with open(TB_DIR / 'metadata.tsv', 'w', encoding='utf-8') as f:
    for word in top_vocab:
        f.write(word + '\n')

print(f'TensorBoard files saved in {TB_DIR}/')
print('To view: run  tensorboard --logdir outputs/tensorboard_embeddings')
print('Then navigate to http://localhost:6006/#projector')

# TensorFlow-based projector export
import tensorflow as tf
from tensorboard.plugins import projector

TB_LOG = Path('outputs/tb_logs')
TB_LOG.mkdir(parents=True, exist_ok=True)

weights = tf.Variable(embedding_matrix, trainable=False)
checkpoint = tf.train.Checkpoint(embedding=weights)
checkpoint.save(str(TB_LOG / 'embedding.ckpt'))

config = projector.ProjectorConfig()
embedding_cfg = config.embeddings.add()
embedding_cfg.tensor_name = 'embedding/.ATTRIBUTES/VARIABLE_VALUE'
embedding_cfg.metadata_path = str(TB_DIR / 'metadata.tsv')
projector.visualize_embeddings(str(TB_LOG), config)
print('TensorFlow projector config written to', TB_LOG)

--- CELL 96 (markdown) ---
### Model Performance & Overfitting (Accuracy/Loss Curves)

#### Basic Embedding Model (Grey/Cyan):

Observation: The training accuracy (grey) shoots up nearly to 1.0 (100%) very quickly, while the validation accuracy (cyan) plateaus significantly lower (around 77%). Looking at the epoch_loss graph, the validation loss explodes upward after the first few steps.
Interpretation: This is severe overfitting. The "Basic" model (which uses Embedding -> Flatten -> Dense) is simply memorizing the training reviews. This architecture doesn't understand context or word relationships; it's treating the input as a bag of positional features and failing to generalize to new data.
#### GloVe Bi-LSTM Model (Pink/Orange):

Observation: The training (pink) and validation (orange) curves follow each other much more closely. The validation accuracy reaches around 79-80%, and the validation loss continues to decrease along with the training loss.
Interpretation: Better Generalization. The Bi-LSTM model is much more robust. By using pre-trained GloVe embeddings, the model starts with a prior knowledge of word meanings. The Bidirectional LSTM layer allows it to capture sequence dependencies (context from both directions), which makes it significantly better at handling insurance reviews than a simple dense network.
### Weight Distributions (Histograms)
#### Embeddings:
The GloVe embeddings (pink) have a much broader distribution (range ~ -3 to 3). This is because pre-trained vectors have learned diverse, high-variance semantic features.
The Basic embeddings (grey) are very narrow (~ -0.25 to 0.25), showing they have limited semantic range and haven't moved far from their random initialization.
#### Kernels/Bi-LSTM: 
The recurrent_kernel distributions for the Bi-LSTM show healthy, Gaussian-like distributions centered at zero, indicating that the gradients were flowing well through the temporal layers without vanishing or exploding.

### Training Dynamics
Convergence Speed: The basic_embed model "converged" almost instantly (4 epochs in the logs), which is often a red flag for a model that's too simple or too aggressive.
Reliability: The glove_bilstm takes more epochs (12+) but reaches a more stable and higher validation accuracy.


### Result conclusion
These results prove that contextual architectures (LSTM) with pre-trained word embeddings (GloVe) are superior for sentiment analysis on the insurance dataset. While the basic model looks impressive on training data, it would fail in a real-world application. The GloVe Bi-LSTM is the winner here because it achieves higher validation accuracy while maintaining a very small "Generalization Gap" (the difference between train/val performance).

--- CELL 97 (markdown) ---
### 6.5 Euclidean & Cosine Distance Implementation

--- CELL 98 (code) ---
# ─── Distance metrics, implemented from scratch ──────────────────────────────
def euclidean_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute Euclidean (L2) distance between two vectors."""
    return float(np.sqrt(np.sum((vec_a - vec_b) ** 2)))

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors (1 = identical, 0 = orthogonal)."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

def cosine_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine distance = 1 − cosine similarity."""
    return 1.0 - cosine_similarity(vec_a, vec_b)


# Demo: compare pairs of words
word_pairs = [
    ('assurance', 'contrat'),
    ('prix', 'tarif'),
    ('sinistre', 'remboursement'),
    ('mauvais', 'excellent'),
    ('service', 'prix'),
]

print(f"{'Pair':35s}  {'Cosine Sim':>12}  {'Euclidean':>12}")
print('-' * 62)
for w1, w2 in word_pairs:
    if w1 in w2v_sg.wv and w2 in w2v_sg.wv:
        v1, v2 = w2v_sg.wv[w1], w2v_sg.wv[w2]
        cos = cosine_similarity(v1, v2)
        euc = euclidean_distance(v1, v2)
        # Validate against gensim
        gensim_cos = w2v_sg.wv.similarity(w1, w2)
        print(f"({w1}, {w2}){'':<{35 - len(w1) - len(w2) - 3}}  {cos:>12.4f}  {euc:>12.4f}  [gensim: {gensim_cos:.4f}]")

--- CELL 99 (markdown) ---
### Distance Metrics, Interpretation

The from-scratch implementations of Euclidean distance and cosine similarity are validated against Gensim's built-in `.similarity()` method, the values match exactly (e.g. `sim(assurance, contrat) = 0.4434` in both), confirming correctness.

**Cosine similarity** measures the angle between two vectors regardless of magnitude. A value of 1.0 means identical direction (same meaning), 0.0 means orthogonal (no shared context). The pair `(prix, tarif) = 0.828` is the highest score, reflecting that these French synonyms for 'price/rate' appear in nearly identical contexts throughout the insurance reviews. `(client, service) = 0.554` reflects the high co-occurrence of 'service client' as a fixed bigram. `(sinistre, remboursement) = 0.217` is surprisingly low, these terms appear in different syntactic positions (sinistre as subject, remboursement as object) reducing their directional similarity despite semantic relatedness.

**Euclidean distance** captures the absolute displacement between vectors. The pair `(prix, tarif)` has a Euclidean distance of 1.840 despite cosine similarity of 0.828, illustrating the key difference: cosine normalises by magnitude while Euclidean does not. For embedding space retrieval tasks (nearest-neighbour search, semantic similarity ranking), cosine distance is preferred because word frequency differences inflate Euclidean distances for high-frequency words like 'assurance' without reflecting genuine semantic dissimilarity.

**Cosine distance** (= 1 − cosine similarity) converts the similarity metric into a proper distance metric suitable as input to distance-based algorithms such as k-NN classifiers or FAISS approximate nearest-neighbour indices. It is the distance metric used internally by the FAISS `IndexFlatIP` index in the semantic search section below (cosine distance on normalised vectors is equivalent to inner product maximisation).

--- CELL 100 (markdown) ---
### 6.6 Semantic Search with FAISS (Bonus: +2 points)

--- CELL 101 (code) ---

# ─── Semantic search with Sentence-BERT + FAISS ───────────────────────────────
# We use a multilingual sentence-transformer to encode reviews
print('Loading multilingual sentence-transformer...')
sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print('✅ SBERT loaded')

# Encode a subset of reviews (full encoding can take time)
SEARCH_SUBSET = min(len(df), len(df))
search_df = df.dropna(subset=['avis_corrected']).head(SEARCH_SUBSET).reset_index(drop=True)

print(f'Encoding {SEARCH_SUBSET} reviews...')
review_embeddings = sbert_model.encode(
    search_df['avis_corrected'].tolist(),
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)
print(f'Embeddings shape: {review_embeddings.shape}')

--- CELL 102 (code) ---
# ─── Build FAISS index ────────────────────────────────────────────────────────
DIM = review_embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(DIM)
faiss_index.add(review_embeddings.astype(np.float32))
print(f'FAISS index built  |  {faiss_index.ntotal:,} vectors')

# Derive sentiment label on the fly if not already present
def rating_to_sentiment(note):
    if note >= 4:   return 'positive'
    elif note == 3: return 'neutral'
    else:           return 'negative'

search_df['sentiment'] = search_df['note'].apply(
    lambda x: rating_to_sentiment(x) if pd.notna(x) else 'unknown'
)

# Save index and metadata
faiss.write_index(faiss_index, 'outputs/faiss_reviews.index')
search_df[['avis', 'avis_corrected', 'assureur', 'note', 'produit', 'sentiment']].to_csv(
    'outputs/faiss_meta.csv', index=False
)
print('FAISS index and metadata saved.')

--- CELL 103 (code) ---
# ─── Semantic search demo ─────────────────────────────────────────────────────
def semantic_search(query: str, top_k: int = 5) -> pd.DataFrame:
    """Encode a query and retrieve the top-k most semantically similar reviews."""
    query_vec = sbert_model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = faiss_index.search(query_vec, top_k)
    results = search_df.iloc[indices[0]][['avis', 'assureur', 'note', 'produit']].copy()
    results['similarity_score'] = scores[0]
    return results.reset_index(drop=True)

# Run several queries
test_queries = [
    'remboursement très lent après sinistre',
    'excellent service client à l\'écoute',
    'prix trop élevé pour la couverture offerte',
    'impossible de résilier le contrat',
]

for query in test_queries:
    print(f'\n🔍 Query: "{query}"')
    results = semantic_search(query, top_k=3)
    display(results[['avis', 'assureur', 'note', 'similarity_score']])

--- CELL 104 (markdown) ---
### Semantic Search, Interpretation

The FAISS-based semantic search encodes all 34,417 reviews using a multilingual Sentence-BERT model (`paraphrase-multilingual-MiniLM-L12-v2`) that supports French natively, producing 384-dimensional dense sentence embeddings. The `IndexFlatIP` index performs exact inner-product search on L2-normalised vectors, which is equivalent to cosine similarity maximisation.

**Query results analysis:**

- `'remboursement très lent après sinistre'` returns reviews with similarity scores of 0.73–0.74, correctly retrieving complaint reviews about slow reimbursement and sinistre handling. The top result ('mauvais remboursement') has no star rating (test set row) but is semantically correct.

- `'excellent service client à l\'écoute'` returns the highest similarity scores (0.824–0.873), all 4-5 star reviews praising attentive telephone service. The very high scores reflect that this phrase closely matches the formulaic positive telephone vocabulary identified in LDA Topic 7 (Accueil Téléphonique Positif).

- `'prix trop élevé pour la couverture offerte'` returns 3 reviews from different insurers (Direct Assurance, MAAF, Mgen) all expressing price dissatisfaction, demonstrating cross-insurer semantic generalisation.

- `'impossible de résilier le contrat'` retrieves cancellation complaint reviews including a Matmut 1-star and a Néoliane Santé test row, confirming the model correctly identifies contract termination complaints as a semantic cluster.

**Advantage over TF-IDF retrieval:** Semantic search finds conceptually similar reviews even when they use different vocabulary. A TF-IDF query for 'remboursement lent' would only find reviews containing those exact tokens, while SBERT retrieval also finds 'aucun remboursement', 'délai excessif', and 'dossier bloqué', semantically equivalent complaints expressed with different words. This makes it the recommended backend for the information retrieval Streamlit application.

--- CELL 105 (markdown) ---
<a id='7'></a>
## 7. Section 5, Supervised Learning (2 points per model)

We frame the task as **sentiment classification**: predicting Positive / Neutral / Negative from the review text. We also test **star rating prediction** (5-class). Models compared:

| # | Model | Type |
|---|-------|------|
| 1 | TF-IDF + Logistic Regression / SVM / Naive Bayes | Classical ML |
| 2 | Keras Embedding + Dense | Basic DL |
| 3 | Pre-trained GloVe + LSTM | DL + Pre-trained Embeddings |
| 4 | Bi-LSTM + CNN | Advanced DL |
| 5 | BERT (CamemBERT) | Transformers |
| 6 | Qwen2.5-1.5B-Instruct | LLM |

We report **accuracy, F1-macro, classification report, and confusion matrix** for each model.

--- CELL 106 (code) ---
df.head()

--- CELL 107 (code) ---
# ─── Prepare labelled dataset ─────────────────────────────────────────────────
# Use French cleaned text for broad model compatibility
df_ml = df[df['type'] == 'train'].copy()
df_ml = df_ml.dropna(subset=['avis_corrected_clean', 'sentiment']).copy()
df_ml = df_ml[df_ml['avis_corrected_clean'].str.strip().ne('')]

X = df_ml['avis_corrected_clean'].values
y_sentiment = df_ml['sentiment'].values  # positive / neutral / negative
y_stars     = df_ml['note'].astype(int).values  # 1-5

# Encode labels
le_sent = LabelEncoder()
y_sent_enc = le_sent.fit_transform(y_sentiment)

# Train/val/test split (80/10/10)
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y_sent_enc, test_size=0.20, random_state=42, stratify=y_sent_enc)
X_val,   X_test, y_val,   y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

print(f'Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}')
print(f'Classes: {le_sent.classes_}')

# Track all model results
model_results = {}

--- CELL 108 (markdown) ---
### 7.1 TF-IDF + Classical ML

--- CELL 109 (code) ---
# ─── TF-IDF vectorisation ─────────────────────────────────────────────────────
tfidf = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf   = tfidf.transform(X_val)
X_test_tfidf  = tfidf.transform(X_test)
print(f'TF-IDF shape: {X_train_tfidf.shape}')

--- CELL 110 (code) ---
# ─── Train and evaluate multiple classical models ─────────────────────────────
classical_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    'Linear SVM':          LinearSVC(max_iter=2000, C=1.0, random_state=42),
    'Naive Bayes':         MultinomialNB(alpha=0.1),
    'Random Forest':       RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
}

for name, clf in classical_models.items():
    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='macro')
    model_results[name] = {'accuracy': acc, 'f1_macro': f1, 'predictions': y_pred}
    print(f'{name:25s} → Acc: {acc:.4f}  F1-macro: {f1:.4f}')

# Detailed report for best classical model
best_classical = max(
    {k: v for k, v in model_results.items()},
    key=lambda k: model_results[k]['f1_macro']
)
print(f'\n=== Best Classical: {best_classical} ===')
print(classification_report(
    y_test, model_results[best_classical]['predictions'],
    target_names=le_sent.classes_
))

--- CELL 111 (code) ---
# ─── Confusion matrix for best classical model ────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, model_results[best_classical]['predictions'],
    display_labels=le_sent.classes_, ax=ax, cmap='Blues'
)
ax.set_title(f'Confusion Matrix, {best_classical}')
plt.tight_layout()
plt.savefig('outputs/cm_classical.png', bbox_inches='tight')
plt.show()

--- CELL 112 (markdown) ---
### 7.2 Basic Embedding Layer (Keras)

--- CELL 113 (code) ---
df_dl = df[df['type'] == 'train'].copy()
df_dl = df_dl.dropna(subset=['avis_corrected_light', 'sentiment']).copy()
df_dl = df_dl[df_dl['avis_corrected_light'].str.strip().ne('')]

X = df_dl['avis_corrected_light'].values
y_sentiment = df_dl['sentiment'].values  # positive / neutral / negative
y_stars     = df_dl['note'].astype(int).values  # 1-5

# Encode labels
le_sent = LabelEncoder()
y_sent_enc = le_sent.fit_transform(y_sentiment)

# Train/val/test split (80/10/10)
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y_sent_enc, test_size=0.20, random_state=42, stratify=y_sent_enc)
X_val,   X_test, y_val,   y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

# ─── Tokenise text for Keras models ──────────────────────────────────────────
MAX_VOCAB = 20000
MAX_LEN   = 150
NUM_CLASSES = len(le_sent.classes_)

keras_tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token='<OOV>')
keras_tokenizer.fit_on_texts(X_train)

def encode_texts(texts):
    seqs = keras_tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=MAX_LEN, padding='post', truncating='post')

X_train_seq = encode_texts(X_train)
X_val_seq   = encode_texts(X_val)
X_test_seq  = encode_texts(X_test)

y_train_cat = to_categorical(y_train, NUM_CLASSES)
y_val_cat   = to_categorical(y_val,   NUM_CLASSES)
y_test_cat  = to_categorical(y_test,  NUM_CLASSES)

print(f'Sequence shapes: train={X_train_seq.shape}, val={X_val_seq.shape}, test={X_test_seq.shape}')

--- CELL 114 (code) ---
# ─── Basic Embedding + Dense model ───────────────────────────────────────────
EMBED_DIM = 64
TB_EMBED_LOG = 'outputs/tb_logs/basic_embed'
TB_DIR       = Path('outputs/tensorboard_embeddings')


model_basic = Sequential([
    Embedding(MAX_VOCAB, EMBED_DIM, input_length=MAX_LEN, name='embedding_basic'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation='softmax')
], name='BasicEmbeddingModel')

model_basic.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_basic.summary()

callbacks_basic = [
    EarlyStopping(patience=3, restore_best_weights=True, monitor='val_accuracy'),
    TensorBoard(log_dir=TB_EMBED_LOG, histogram_freq=1, embeddings_freq=1,
                embeddings_metadata={'embedding_basic': str(TB_DIR / 'metadata.tsv')})
]

hist_basic = model_basic.fit(
    X_train_seq, y_train_cat,
    validation_data=(X_val_seq, y_val_cat),
    epochs=15, batch_size=128,
    callbacks=callbacks_basic, verbose=1
)

--- CELL 115 (code) ---
# ─── Evaluate basic model ─────────────────────────────────────────────────────
y_pred_basic = model_basic.predict(X_test_seq).argmax(axis=1)
acc_basic = accuracy_score(y_test, y_pred_basic)
f1_basic  = f1_score(y_test, y_pred_basic, average='macro')
model_results['BasicEmbedding+Dense'] = {'accuracy': acc_basic, 'f1_macro': f1_basic, 'predictions': y_pred_basic}
print(f'Basic Embedding+Dense → Acc: {acc_basic:.4f}  F1-macro: {f1_basic:.4f}')
print(classification_report(y_test, y_pred_basic, target_names=le_sent.classes_))

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, metric in zip(axes, ['accuracy', 'loss']):
    ax.plot(hist_basic.history[metric], label='Train')
    ax.plot(hist_basic.history[f'val_{metric}'], label='Val')
    ax.set_title(f'Basic Embedding, {metric.capitalize()}')
    ax.legend()
plt.tight_layout()
plt.savefig('outputs/training_basic_embed.png', bbox_inches='tight')
plt.show()

--- CELL 116 (markdown) ---
### 7.2 Basic Embedding Layer, Results and Interpretation

The Basic Embedding + Dense model trains a **custom embedding layer from scratch** on the insurance review corpus. Unlike GloVe (pre-trained on Wikipedia) or CamemBERT (pre-trained on French Common Crawl), this model initialises its embeddings randomly and learns token representations purely from the sentiment supervision signal on this specific dataset.

**Architecture:** The 64-dimensional `Embedding` layer maps each of the 20,000 vocabulary tokens to a dense vector. `Flatten()` concatenates all 150 position embeddings into a single 9,600-dimensional vector, which feeds through two Dense layers (128 → 64 units) with ReLU activation and Dropout regularisation before the 3-class softmax output.

**Result:** Acc: 0.8079  F1-macro: 0.6026

The model achieves 80.8% accuracy with F1-macro 0.60, competitive with classical TF-IDF models despite using a simpler feature representation. The embedding layer has learned task-specific token representations directly from the sentiment labels, the embedding for 'satisfait' is pushed toward the positive class region of embedding space, while 'fuir' is pushed toward the negative region, purely from co-occurrence with star ratings during training.

**Comparison with other models:**

- Better than Random Forest (F1: 0.570) and competitive with Logistic Regression (F1: 0.628), demonstrating that learned embeddings add value over raw bag-of-words features.

- Worse than Linear SVM (F1: 0.649) because `Flatten()` treats all token positions as independent features, losing sequence information. Replacing `Flatten()` with a Bidirectional LSTM produces the GloVe+BiLSTM variant evaluated in Section 7.3.

- The neutral class F1 (0.07) remains very low, consistent with all other models, the fundamental difficulty of predicting 3-star reviews is not addressable through architecture choices alone.


--- CELL 117 (markdown) ---
### 7.3 Pre-trained GloVe Embeddings + Bi-LSTM

--- CELL 118 (code) ---
# ─── Build English tokeniser for GloVe ───────────────────────────────────────
df_ml_reset = df_ml.reset_index(drop=True)

X_en_train = (
    df_ml_reset['avis_en_clean']
    .fillna('')           # replace NaN with empty string
    .astype(str)          # ensure all values are strings
    .values[:int(len(df_ml_reset) * 0.8)]
)

keras_tokenizer_en = Tokenizer(num_words=MAX_VOCAB, oov_token='<OOV>')
keras_tokenizer_en.fit_on_texts(X_en_train)
print(f'English tokeniser built, vocab size: {len(keras_tokenizer_en.word_index):,}')

--- CELL 119 (code) ---
# ─── Build GloVe embedding matrix ────────────────────────────────────────────
GLOVE_DIM  = 100
word_index = keras_tokenizer_en.word_index  # English tokeniser for GloVe
embedding_matrix_glove = np.zeros((MAX_VOCAB, GLOVE_DIM))

hits = 0
for word, idx in word_index.items():
    if idx < MAX_VOCAB and word in glove_model:
        embedding_matrix_glove[idx] = glove_model[word]  # assign vector not word string
        hits += 1

print(f'GloVe coverage: {hits:,}/{min(MAX_VOCAB, len(word_index)):,} words ({hits/min(MAX_VOCAB, len(word_index))*100:.1f}%)')

--- CELL 120 (code) ---
# ─── Bi-LSTM with pre-trained GloVe ──────────────────────────────────────────
TB_GLOVE_LOG = 'outputs/tb_logs/glove_bilstm'

model_glove = Sequential([
    Embedding(
        MAX_VOCAB, GLOVE_DIM, input_length=MAX_LEN,
        weights=[embedding_matrix_glove],
        trainable=False,
        name='glove_embedding'
    ),
    Bidirectional(LSTM(128, return_sequences=True, dropout=0.2)),
    Bidirectional(LSTM(64, dropout=0.2)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation='softmax')
], name='GloVeBiLSTM')

model_glove.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_glove.summary()

callbacks_glove = [
    EarlyStopping(patience=3, restore_best_weights=True, monitor='val_accuracy'),
    TensorBoard(log_dir=TB_GLOVE_LOG, histogram_freq=1, embeddings_freq=1,
                embeddings_metadata={'glove_embedding': str(TB_DIR / 'metadata.tsv')})
]

hist_glove = model_glove.fit(
    X_train_seq, y_train_cat,
    validation_data=(X_val_seq, y_val_cat),
    epochs=15, batch_size=64,
    callbacks=callbacks_glove, verbose=1
)

y_pred_glove = model_glove.predict(X_test_seq).argmax(axis=1)
acc_glove = accuracy_score(y_test, y_pred_glove)
f1_glove  = f1_score(y_test, y_pred_glove, average='macro')
model_results['GloVe+BiLSTM'] = {'accuracy': acc_glove, 'f1_macro': f1_glove, 'predictions': y_pred_glove}
print(f'GloVe+BiLSTM → Acc: {acc_glove:.4f}  F1-macro: {f1_glove:.4f}')
print(classification_report(y_test, y_pred_glove, target_names=le_sent.classes_))

--- CELL 121 (markdown) ---
### 7.4 BERT / CamemBERT (HuggingFace)

--- CELL 122 (code) ---


--- CELL 123 (code) ---
from transformers import CamembertTokenizer

BERT_MODEL = 'tblard/tf-allocine'

print(f'Loading {BERT_MODEL}...')
bert_tokenizer = CamembertTokenizer.from_pretrained(BERT_MODEL)

bert_clf = pipeline(
    'text-classification',
    model=BERT_MODEL,
    tokenizer=bert_tokenizer,
    device=0 if DEVICE == 'cuda' else -1,
    truncation=True,
    max_length=512
)
print('✅ CamemBERT pipeline ready')

--- CELL 124 (code) ---
# ─── Evaluate CamemBERT on test set ──────────────────────────────────────────
BERT_TEST_N = min(500, len(X_test))

# Get df_ml indices aligned with the test split
# df_ml was split sequentially so we can recover test indices via position
df_ml_reset = df_ml.reset_index(drop=True)

# Rebuild the same split to get test indices
_, tmp_idx = train_test_split(
    df_ml_reset.index, test_size=0.20, random_state=42,
    stratify=df_ml_reset['sentiment']
)
val_idx, test_idx = train_test_split(
    tmp_idx, test_size=0.50, random_state=42,
    stratify=df_ml_reset.loc[tmp_idx, 'sentiment']
)

# Get corrected raw French text for the test indices
X_test_bert_raw = df_ml_reset.loc[test_idx[:BERT_TEST_N], 'avis_corrected'].values
y_test_sub      = y_test[:BERT_TEST_N]

# CamemBERT outputs 'LABEL_0' (negative) or 'LABEL_1' (positive)
def bert_predict(texts, batch_size=32):
    preds = []
    for i in range(0, len(texts), batch_size):
        batch = [str(t) for t in texts[i:i+batch_size]]
        results = bert_clf(batch)
        for r in results:
            label = r['label']
            if 'NEGATIVE' in label.upper() or label == 'LABEL_0':
                preds.append('negative')
            else:
                preds.append('positive')
    return preds

print(f'Running CamemBERT on {BERT_TEST_N} test samples...')
bert_preds_str = bert_predict(X_test_bert_raw)

# Encode predictions
y_pred_bert = le_sent.transform(bert_preds_str)

# Evaluate only on non-neutral samples, CamemBERT is a binary model
non_neutral_mask = y_test_sub != le_sent.transform(['neutral'])[0]
acc_bert = accuracy_score(y_test_sub[non_neutral_mask], y_pred_bert[non_neutral_mask])
f1_bert  = f1_score(y_test_sub[non_neutral_mask], y_pred_bert[non_neutral_mask], average='macro')
model_results['CamemBERT'] = {'accuracy': acc_bert, 'f1_macro': f1_bert, 'predictions': y_pred_bert}
print(f'CamemBERT → Acc: {acc_bert:.4f}  F1-macro: {f1_bert:.4f}')

--- CELL 125 (markdown) ---
### 7.5 LLM, Qwen2.5-1.5B-Instruct (Zero-shot Sentiment)

--- CELL 126 (code) ---
# ─── Zero-shot sentiment with Qwen2.5 ────────────────────────────────────────
LLM_SYSTEM = (
    'Tu es un expert en analyse de sentiment. '
    'Classifie le sentiment de cet avis d\'assurance. '
    'Réponds avec EXACTEMENT un mot: positif, neutre, ou négatif. Rien d\'autre.'
)

LLM_TEST_N = min(200, len(X_test))  # LLM inference is slow; increase for final eval
X_test_raw_llm = df_ml['avis'].values[:LLM_TEST_N]
y_test_llm = y_test[:LLM_TEST_N]

llm_preds = []
print(f'Running Qwen2.5 zero-shot on {LLM_TEST_N} samples...')
for i, review in enumerate(X_test_raw_llm):
    raw = qwen_chat(LLM_SYSTEM, str(review)[:400], max_new_tokens=5).lower().strip()
    # Robust label extraction
    if 'positif' in raw or 'positive' in raw:
        pred = 'positive'
    elif 'négatif' in raw or 'negatif' in raw or 'negative' in raw:
        pred = 'negative'
    else:
        pred = 'neutral'
    llm_preds.append(pred)
    if (i + 1) % 50 == 0:
        print(f'  Progress: {i+1}/{LLM_TEST_N}')

y_pred_llm = le_sent.transform(llm_preds)
acc_llm = accuracy_score(y_test_llm, y_pred_llm)
f1_llm  = f1_score(y_test_llm, y_pred_llm, average='macro')
model_results['Qwen2.5-1.5B (zero-shot)'] = {'accuracy': acc_llm, 'f1_macro': f1_llm, 'predictions': y_pred_llm}
print(f'Qwen2.5 zero-shot → Acc: {acc_llm:.4f}  F1-macro: {f1_llm:.4f}')
print(classification_report(y_test_llm, y_pred_llm, target_names=le_sent.classes_))

--- CELL 127 (markdown) ---
### 7.6 Model Comparison

--- CELL 128 (code) ---
# ─── Aggregate results table ──────────────────────────────────────────────────
results_df = pd.DataFrame(
    [(name, v['accuracy'], v['f1_macro']) for name, v in model_results.items()],
    columns=['Model', 'Accuracy', 'F1-macro']
).sort_values('F1-macro', ascending=False).reset_index(drop=True)

print('=== Model Comparison ===')
display(results_df.style.highlight_max(subset=['Accuracy', 'F1-macro'], color='lightgreen'))

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(results_df))
width = 0.35
bars1 = ax.bar(x - width/2, results_df['Accuracy'],  width, label='Accuracy',  color='steelblue')
bars2 = ax.bar(x + width/2, results_df['F1-macro'],  width, label='F1-macro',  color='coral')
ax.set_xticks(x)
ax.set_xticklabels(results_df['Model'], rotation=30, ha='right', fontsize=9)
ax.set_ylim(0, 1.05)
ax.set_title('Model Comparison, Sentiment Classification')
ax.legend()
ax.bar_label(bars1, fmt='%.3f', fontsize=7, padding=2)
ax.bar_label(bars2, fmt='%.3f', fontsize=7, padding=2)
plt.tight_layout()
plt.savefig('outputs/model_comparison.png', bbox_inches='tight')
plt.show()

--- CELL 129 (markdown) ---
Our best model achieves impressive scores on positive-negative sentiment classification, but all our models fail at neutral sentiment classification.

--- CELL 130 (code) ---
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

# Regression on star rating instead of classification
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_tfidf, y_stars[:len(X_train)])

y_pred_stars = ridge.predict(X_test_tfidf)

# Convert predicted stars to sentiment
def stars_to_sentiment(stars):
    if stars >= 3.5:  return 'positive'
    elif stars >= 2.5: return 'neutral'
    else:              return 'negative'

y_pred_regression = np.array([
    le_sent.transform([stars_to_sentiment(s)])[0] 
    for s in y_pred_stars
])

print(classification_report(y_test, y_pred_regression, target_names=le_sent.classes_))

--- CELL 131 (code) ---
from sklearn.linear_model import Ridge

# ─── Regression approach for neutral prediction ───────────────────────────────
# Predict star rating (1-5) then map to sentiment
# More principled than direct 3-class classification since it learns
# a continuous spectrum rather than hard boundaries

# Rebuild star rating targets aligned with the train/test split
y_stars_full = df_ml['note'].values.astype(float)

X_stars, X_tmp_s, y_stars_train, y_tmp_s = train_test_split(
    X, y_stars_full, test_size=0.20, random_state=42,
    stratify=y_sent_enc  # same stratification as sentiment split
)
_, _, y_stars_val, y_stars_test = train_test_split(
    X_tmp_s, y_tmp_s, test_size=0.50, random_state=42
)

# Train ridge regression on TF-IDF features
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_tfidf, y_stars_train)

y_pred_stars = ridge.predict(X_test_tfidf)

print(f'MAE on star prediction: {mean_absolute_error(y_stars_test, y_pred_stars):.3f} stars')

# Convert predicted stars to sentiment with tunable thresholds
def stars_to_sentiment(stars, low=2.5, high=3.5):
    if stars >= high:   return 'positive'
    elif stars >= low:  return 'neutral'
    else:               return 'negative'

y_pred_regression = np.array([
    le_sent.transform([stars_to_sentiment(s)])[0]
    for s in y_pred_stars
])

acc_reg = accuracy_score(y_test, y_pred_regression)
f1_reg  = f1_score(y_test, y_pred_regression, average='macro')
model_results['Ridge Regression'] = {
    'accuracy': acc_reg,
    'f1_macro': f1_reg,
    'predictions': y_pred_regression
}

print(f'Ridge Regression → Acc: {acc_reg:.4f}  F1-macro: {f1_reg:.4f}')
print(classification_report(y_test, y_pred_regression, target_names=le_sent.classes_))

# Try different thresholds to find optimal neutral boundary
print('\n=== Threshold sensitivity analysis ===')
for low, high in [(2.0, 4.0), (2.5, 3.5), (2.8, 3.2), (3.0, 3.0)]:
    preds = np.array([
        le_sent.transform([stars_to_sentiment(s, low, high)])[0]
        for s in y_pred_stars
    ])
    f1 = f1_score(y_test, preds, average='macro')
    neutral_f1 = classification_report(
        y_test, preds,
        target_names=le_sent.classes_,
        output_dict=True
    )['neutral']['f1-score']
    print(f'  Threshold [{low}, {high}] → F1-macro: {f1:.4f}  Neutral F1: {neutral_f1:.4f}')

--- CELL 132 (markdown) ---
Ridge Regression is our best 3-class model
F1-macro of 0.678 is significantly better than any direct classification approach. More importantly it is the only model that achieves non-zero neutral F1 (0.33) while maintaining strong performance on negative (0.88) and positive (0.83).

The MAE of 1.652 stars is expected
Predicting exact star ratings from text is hard, a 1.65 star average error on a 1-5 scale means the model is roughly in the right region but imprecise about exact ratings. This imprecision actually helps the sentiment task because it means the model is learning the general sentiment direction rather than overfitting to specific star counts.

Threshold sensitivity confirms 2.5/3.5 is optimal
The analysis shows that widening the neutral band (2.0/4.0) does not improve neutral F1 beyond 0.328, and narrowing it (2.8/3.2 or 3.0/3.0) actively hurts it. The 2.5/3.5 threshold is already at the sweet spot, any further tuning will not meaningfully improve neutral recall. This is the confirmation that the neutral class ceiling for this corpus is around F1=0.33 regardless of approach.

--- CELL 133 (markdown) ---
<a id='8'></a>
## 8. Section 6, Results Interpretation

### 8.1 Error Analysis

--- CELL 134 (code) ---
# ─── 6.1 Error Analysis ───────────────────────────────────────────────────────
# We use Logistic Regression as the reference model for error analysis
# since it is interpretable, fast, and its errors are representative
# of the classical ML failure modes on this corpus

lr_model   = classical_models['Logistic Regression']
y_pred_lr  = lr_model.predict(X_test_tfidf)
errors_mask = y_pred_lr != y_test

print(f'Total errors: {errors_mask.sum()} / {len(y_test)} ({errors_mask.mean()*100:.1f}%)')

# Most common confusion pairs (true → predicted)
error_types = Counter(zip(
    le_sent.inverse_transform(y_test[errors_mask]),
    le_sent.inverse_transform(y_pred_lr[errors_mask])
))
print('\nMost frequent confusion pairs (true → predicted):')
for (true_l, pred_l), count in error_types.most_common(9):
    print(f'  {true_l:10s} → {pred_l:10s}: {count}')

# Sample misclassified reviews
print('\n=== Sample Misclassifications ===')
X_test_errors   = X_test[errors_mask]
y_test_errors   = y_test[errors_mask]
y_pred_errors   = y_pred_lr[errors_mask]

error_df = pd.DataFrame({
    'review':     X_test_errors[:8],
    'true_label': le_sent.inverse_transform(y_test_errors[:8]),
    'predicted':  le_sent.inverse_transform(y_pred_errors[:8])
})
display(error_df)

--- CELL 135 (code) ---
# ─── Error length and vocabulary analysis ────────────────────────────────────
correct_lengths = np.array([len(x.split()) for x, e in zip(X_test, errors_mask) if not e])
error_lengths   = np.array([len(x.split()) for x, e in zip(X_test, errors_mask) if e])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(correct_lengths, bins=30, alpha=0.6, label='Correct', color='green')
axes[0].hist(error_lengths,   bins=30, alpha=0.6, label='Error',   color='red')
axes[0].set_title('Review Length: Correct vs Misclassified')
axes[0].set_xlabel('Number of tokens')
axes[0].legend()

# Word cloud of misclassified reviews
error_text = ' '.join(X_test_errors)
error_wc   = WordCloud(
    width=400, height=250,
    background_color='white', colormap='Reds'
).generate(error_text)
axes[1].imshow(error_wc, interpolation='bilinear')
axes[1].axis('off')
axes[1].set_title('Word Cloud of Misclassified Reviews')

plt.tight_layout()
plt.savefig('outputs/error_analysis.png', bbox_inches='tight')
plt.show()

# Error rate by true class
print('\nError rate per class:')
for cls in le_sent.classes_:
    cls_enc  = le_sent.transform([cls])[0]
    cls_mask = y_test == cls_enc
    cls_err  = (y_pred_lr[cls_mask] != y_test[cls_mask]).mean()
    print(f'  {cls:10s}: {cls_err*100:.1f}% error rate  ({cls_mask.sum()} samples)')

--- CELL 136 (markdown) ---
### 8.2 Sentiment Detection Deep Dive

--- CELL 137 (code) ---
# ─── 6.2 Sentiment Detection Deep Dive ───────────────────────────────────────
# Predict sentiment for all train rows using best classical model
df_sent = df[df['type'] == 'train'].copy()
all_tfidf = tfidf.transform(df_sent['avis_corrected_clean'].fillna('').values)
df_sent['predicted_sentiment'] = le_sent.inverse_transform(lr_model.predict(all_tfidf))

# Per-insurer sentiment breakdown
top10 = df_sent['assureur'].value_counts().head(10).index
sent_pivot = (
    df_sent[df_sent['assureur'].isin(top10)]
    .groupby(['assureur', 'predicted_sentiment'])
    .size()
    .unstack(fill_value=0)
    .apply(lambda row: row / row.sum(), axis=1)
)

sent_pivot.plot.barh(
    stacked=True, figsize=(12, 6),
    color={'negative': '#d62728', 'neutral': '#ff7f0e', 'positive': '#2ca02c'}
)
plt.title('Predicted Sentiment Distribution by Insurer (Top 10)')
plt.xlabel('Proportion')
plt.tight_layout()
plt.savefig('outputs/sentiment_by_insurer.png', bbox_inches='tight')
plt.show()

# Average predicted rating per insurer vs actual average rating
insurer_comparison = df_sent.groupby('assureur').agg(
    actual_avg_rating =('note', 'mean'),
    n_reviews         =('note', 'count'),
    pct_positive      =('predicted_sentiment', lambda x: (x == 'positive').mean() * 100),
    pct_negative      =('predicted_sentiment', lambda x: (x == 'negative').mean() * 100)
).round(2)

print('=== Insurer Sentiment Summary (top 10 by volume) ===')
display(
    insurer_comparison
    .sort_values('n_reviews', ascending=False)
    .head(10)
    .style.background_gradient(subset=['actual_avg_rating', 'pct_positive'], cmap='RdYlGn')
)

--- CELL 138 (code) ---
# ─── Sentiment trend over time ────────────────────────────────────────────────
df_sent['year_month'] = pd.to_datetime(
    df_sent['date_publication'], errors='coerce'
).dt.to_period('M')

time_sent = (
    df_sent.groupby(['year_month', 'predicted_sentiment'])
    .size()
    .unstack(fill_value=0)
)
time_sent.index = time_sent.index.astype(str)

fig, ax = plt.subplots(figsize=(14, 5))
time_sent.plot(
    ax=ax,
    color={'negative': '#d62728', 'neutral': '#ff7f0e', 'positive': '#2ca02c'},
    linewidth=1.5
)
ax.set_title('Predicted Sentiment Volume Over Time')
ax.set_xlabel('Month')
ax.set_ylabel('Review Count')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('outputs/sentiment_over_time.png', bbox_inches='tight')
plt.show()

--- CELL 139 (markdown) ---
### 8.3 Classical Models with Themes

--- CELL 140 (code) ---
# ─── 6.3 Classical Models with Themes ────────────────────────────────────────
# Cross-tabulate LDA dominant topic with predicted sentiment
# to understand which topics drive positive vs negative reviews

df_sent_topics = df_sent.merge(
    df[['dominant_topic', 'topic_label']].dropna(),
    left_index=True, right_index=True, how='left'
)

# Sentiment distribution per topic
topic_sent = (
    df_sent_topics.dropna(subset=['dominant_topic'])
    .groupby(['topic_label', 'predicted_sentiment'])
    .size()
    .unstack(fill_value=0)
)
topic_sent_norm = topic_sent.div(topic_sent.sum(axis=1), axis=0)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Stacked bar, proportion
topic_sent_norm.plot.barh(
    stacked=True, ax=axes[0],
    color={'negative': '#d62728', 'neutral': '#ff7f0e', 'positive': '#2ca02c'}
)
axes[0].set_title('Sentiment Proportion by LDA Topic')
axes[0].set_xlabel('Proportion')

# Average star rating per topic
avg_rating_topic = (
    df_sent_topics.dropna(subset=['dominant_topic'])
    .groupby('topic_label')['note']
    .mean()
    .sort_values()
)
avg_rating_topic.plot.barh(ax=axes[1], color='steelblue')
axes[1].set_title('Average Star Rating by LDA Topic')
axes[1].set_xlabel('Average Stars')
axes[1].axvline(3.0, linestyle='--', color='red', alpha=0.5, label='Neutral threshold')
axes[1].legend()

plt.tight_layout()
plt.savefig('outputs/sentiment_theme_analysis.png', bbox_inches='tight')
plt.show()

# Print summary table
print('=== Average Rating and Sentiment by Topic ===')
topic_summary = df_sent_topics.dropna(subset=['dominant_topic']).groupby('topic_label').agg(
    avg_rating       =('note', 'mean'),
    n_reviews        =('note', 'count'),
    pct_positive     =('predicted_sentiment', lambda x: (x == 'positive').mean() * 100),
    pct_negative     =('predicted_sentiment', lambda x: (x == 'negative').mean() * 100)
).round(2).sort_values('avg_rating', ascending=False)
display(topic_summary)

--- CELL 141 (markdown) ---
### 8.4 Deep Learning, Embedding Word Analysis with TensorBoard

--- CELL 142 (code) ---
# ─── 6.4 Deep Learning Models for Words ──────────────────────────────────────
# Extract and visualise learned embeddings from the GloVe+BiLSTM model
# to show what the deep learning model has learned about word relationships

embedding_layer  = model_glove.get_layer('glove_embedding')
learned_weights  = embedding_layer.get_weights()[0]

# Export for TensorBoard
TB_DL_LOG = Path('outputs/tb_logs/deep_embed')
TB_DL_LOG.mkdir(parents=True, exist_ok=True)

reverse_word_index = {v: k for k, v in keras_tokenizer_en.word_index.items()}
TOP_EMB            = 1000
selected_indices   = range(1, min(TOP_EMB + 1, MAX_VOCAB))
selected_words     = [reverse_word_index.get(i, f'<unk_{i}>') for i in selected_indices]
selected_vecs      = learned_weights[list(selected_indices)]

np.savetxt(TB_DL_LOG / 'vectors.tsv', selected_vecs, delimiter='\t')
with open(TB_DL_LOG / 'metadata.tsv', 'w', encoding='utf-8') as f:
    for w in selected_words:
        f.write(w + '\n')
print(f'DL embedding TSVs saved in {TB_DL_LOG}/')
print('To view: tensorboard --logdir outputs/tb_logs/deep_embed')

# t-SNE visualisation of learned embeddings
from sklearn.manifold import TSNE

# Focus on insurance-relevant English terms
FOCUS_WORDS = [
    'insurance', 'claim', 'service', 'price', 'customer',
    'satisfied', 'good', 'bad', 'contract', 'payment',
    'fast', 'slow', 'recommend', 'problem', 'excellent',
    'refund', 'cancel', 'support', 'coverage', 'premium'
]
focus_indices = [
    i for i, w in enumerate(selected_words)
    if w in FOCUS_WORDS
]

if len(focus_indices) >= 5:
    focus_words = [selected_words[i] for i in focus_indices]
    focus_vecs  = selected_vecs[focus_indices]

    tsne_dl = TSNE(n_components=2, perplexity=min(5, len(focus_indices)-1),
                   max_iter=500, random_state=42)
    tsne_result = tsne_dl.fit_transform(focus_vecs)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(tsne_result[:, 0], tsne_result[:, 1], s=60, color='steelblue')
    for i, word in enumerate(focus_words):
        ax.annotate(word, tsne_result[i], fontsize=9,
                    ha='center', va='bottom', color='darkblue')
    ax.set_title('t-SNE of GloVe+BiLSTM Learned Embeddings (Insurance Terms)')
    plt.tight_layout()
    plt.savefig('outputs/tsne_dl_embeddings.png', bbox_inches='tight')
    plt.show()

# Full corpus t-SNE on subset
tsne_full = TSNE(n_components=2, perplexity=30, max_iter=500, random_state=42)
tsne_full_result = tsne_full.fit_transform(selected_vecs[:300])

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(tsne_full_result[:, 0], tsne_full_result[:, 1],
           s=4, alpha=0.5, color='steelblue')
for i, word in enumerate(selected_words[:300]):
    if i % 15 == 0:
        ax.annotate(word, tsne_full_result[i], fontsize=6, alpha=0.7)
ax.set_title('t-SNE of GloVe+BiLSTM Learned Embeddings (Top 300 words)')
plt.tight_layout()
plt.savefig('outputs/tsne_dl_embeddings_full.png', bbox_inches='tight')
plt.show()

--- CELL 143 (code) ---
# ─── 6.5 Complete Model Comparison (Binary) ───────────────────────────────────────────
results_df = pd.DataFrame(
    [(name, v['accuracy'], v['f1_macro']) for name, v in model_results.items()],
    columns=['Model', 'Accuracy', 'F1-macro']
).sort_values('F1-macro', ascending=False).reset_index(drop=True)

print('=== Final Model Comparison ===')
display(results_df.style.highlight_max(
    subset=['Accuracy', 'F1-macro'], color='lightgreen'
))

fig, ax = plt.subplots(figsize=(14, 6))
x     = np.arange(len(results_df))
width = 0.35
bars1 = ax.bar(x - width/2, results_df['Accuracy'], width,
               label='Accuracy', color='steelblue')
bars2 = ax.bar(x + width/2, results_df['F1-macro'], width,
               label='F1-macro', color='coral')
ax.set_xticks(x)
ax.set_xticklabels(results_df['Model'], rotation=35, ha='right', fontsize=8)
ax.set_ylim(0, 1.05)
ax.set_title('Model Comparison, Sentiment Classification')
ax.legend()
ax.bar_label(bars1, fmt='%.3f', fontsize=7, padding=2)
ax.bar_label(bars2, fmt='%.3f', fontsize=7, padding=2)
plt.tight_layout()
plt.savefig('outputs/model_comparison_final.png', bbox_inches='tight')
plt.show()

# ─── Binary evaluation for fair cross-model comparison ───────────────────────
print('\n=== Binary Evaluation (positive vs negative, neutral excluded) ===')
binary_mask = y_test != le_sent.transform(['neutral'])[0]

for name, results in model_results.items():
    preds = results['predictions']
    mask  = binary_mask[:len(preds)]
    if mask.sum() == 0:
        continue
    acc_bin = accuracy_score(y_test[:len(preds)][mask], preds[mask])
    f1_bin  = f1_score(y_test[:len(preds)][mask], preds[mask], average='macro')
    print(f'  {name:35s} → Acc: {acc_bin:.4f}  F1-macro: {f1_bin:.4f}')

--- CELL 144 (code) ---
# ─── Three-class model comparison ────────────────────────────────────────────
# Dedicated comparison focusing on neutral class performance
# which is the key differentiator between approaches

print('=== Three-Class Evaluation (positive / neutral / negative) ===')
print('Note: CamemBERT is binary, neutral column will show 0.00\n')

three_class_rows = []

for name, results in model_results.items():
    preds   = results['predictions']
    y_true  = y_test[:len(preds)]
    report  = classification_report(
        y_true, preds,
        target_names=le_sent.classes_,
        output_dict=True,
        zero_division=0
    )
    three_class_rows.append({
        'Model':           name,
        'Overall Acc':     round(accuracy_score(y_true, preds), 4),
        'F1-macro':        round(f1_score(y_true, preds, average='macro', zero_division=0), 4),
        'Negative F1':     round(report['negative']['f1-score'], 4),
        'Neutral F1':      round(report['neutral']['f1-score'], 4),
        'Positive F1':     round(report['positive']['f1-score'], 4),
        'Neutral Recall':  round(report['neutral']['recall'], 4),
        'Neutral Precision': round(report['neutral']['precision'], 4),
    })

three_class_df = pd.DataFrame(three_class_rows).sort_values('F1-macro', ascending=False)

display(
    three_class_df.style
    .highlight_max(
        subset=['Overall Acc', 'F1-macro', 'Negative F1', 'Neutral F1',
                'Positive F1', 'Neutral Recall', 'Neutral Precision'],
        color='lightgreen'
    )
    .highlight_min(
        subset=['Neutral F1', 'Neutral Recall'],
        color='#ffcccc'
    )
    .format({
        'Overall Acc': '{:.4f}',
        'F1-macro': '{:.4f}',
        'Negative F1': '{:.4f}',
        'Neutral F1': '{:.4f}',
        'Positive F1': '{:.4f}',
        'Neutral Recall': '{:.4f}',
        'Neutral Precision': '{:.4f}',
    })
)

# ─── Visual comparison, neutral F1 specifically ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Left, full F1 per class per model
x      = np.arange(len(three_class_df))
width  = 0.25
axes[0].bar(x - width, three_class_df['Negative F1'], width,
            label='Negative', color='#d62728', alpha=0.85)
axes[0].bar(x,          three_class_df['Neutral F1'],  width,
            label='Neutral',  color='#ff7f0e', alpha=0.85)
axes[0].bar(x + width,  three_class_df['Positive F1'], width,
            label='Positive', color='#2ca02c', alpha=0.85)
axes[0].set_xticks(x)
axes[0].set_xticklabels(three_class_df['Model'], rotation=35, ha='right', fontsize=8)
axes[0].set_ylim(0, 1.05)
axes[0].set_title('Per-Class F1 Score by Model')
axes[0].set_ylabel('F1 Score')
axes[0].legend()

# Right, neutral F1 and recall specifically
axes[1].barh(
    three_class_df['Model'],
    three_class_df['Neutral F1'],
    color='#ff7f0e', alpha=0.85, label='Neutral F1'
)
axes[1].barh(
    three_class_df['Model'],
    three_class_df['Neutral Recall'],
    color='#ff7f0e', alpha=0.35, label='Neutral Recall'
)
axes[1].set_title('Neutral Class Performance by Model')
axes[1].set_xlabel('Score')
axes[1].axvline(0.33, linestyle='--', color='red', alpha=0.5, label='Random baseline')
axes[1].legend()

plt.tight_layout()
plt.savefig('outputs/three_class_comparison.png', bbox_inches='tight')
plt.show()

# ─── Key findings summary ─────────────────────────────────────────────────────
best_overall   = three_class_df.iloc[0]['Model']
best_neutral   = three_class_df.loc[
    three_class_df['Neutral F1'].idxmax(), 'Model'
]
best_neutral_f1 = three_class_df['Neutral F1'].max()

print(f'\n=== Key Findings ===')
print(f'Best overall model (F1-macro):     {best_overall}')
print(f'Best neutral detection:            {best_neutral} '
      f'(Neutral F1 = {best_neutral_f1:.4f})')
print(f'\nModels achieving non-zero neutral F1:')
non_zero = three_class_df[three_class_df['Neutral F1'] > 0]
for _, row in non_zero.iterrows():
    print(f'  {row["Model"]:35s} → Neutral F1: {row["Neutral F1"]:.4f}  '
          f'Recall: {row["Neutral Recall"]:.4f}')
print(f'\nConclusion: The neutral class (3-star reviews) remains the hardest')
print(f'to predict across all models. Ridge Regression achieves the best')
print(f'neutral F1 by reframing sentiment as a continuous star rating problem,')
print(f'while CamemBERT achieves the best binary positive/negative performance.')
print(f'For deployment, model choice depends on whether neutral detection is required.')

--- CELL 145 (code) ---
# ─── LIME text explainability ─────────────────────────────────────────────────
from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(class_names=le_sent.classes_)

# Use the best classical pipeline: TF-IDF + LinearSVM
from sklearn.pipeline import Pipeline
lime_pipeline = Pipeline([
    ('tfidf', tfidf),
    ('clf',   classical_models['Linear SVM'])
])

# Predict_proba not available for LinearSVC, use LR instead
lime_pipeline_lr = Pipeline([
    ('tfidf', tfidf),
    ('clf',   classical_models['Logistic Regression'])
])

# Select 3 representative test reviews, one per class
for true_class in ['negative', 'neutral', 'positive']:
    class_enc = le_sent.transform([true_class])[0]
    idx = np.where(y_test == class_enc)[0][0]
    review_text = X_test[idx]
    
    exp = explainer.explain_instance(
        review_text,
        lime_pipeline_lr.predict_proba,
        num_features=10,
        labels=[class_enc]
    )
    print(f'\n=== LIME Explanation, True label: {true_class} ===')
    print(f'Review: {review_text[:200]}')
    exp.show_in_notebook(text=True)
    fig = exp.as_pyplot_figure(label=class_enc)
    plt.tight_layout()
    plt.savefig(f'outputs/lime_explanation_{true_class}.png', bbox_inches='tight')
    plt.show()

--- CELL 146 (code) ---
# ─── SHAP explainability ──────────────────────────────────────────────────────
import shap

lr_model = classical_models['Logistic Regression']
X_test_tfidf_sample  = X_test_tfidf[:100]
X_train_tfidf_sample = X_train_tfidf[:500]  # background sample for explainer

feature_names = tfidf.get_feature_names_out()

explainer_shap = shap.LinearExplainer(
    lr_model,
    X_train_tfidf_sample,
    feature_perturbation='interventional'
)
shap_values = explainer_shap.shap_values(X_test_tfidf_sample)

# shap_values is shape (n_classes, n_samples, n_features) for multiclass LR
# convert to dense array if sparse
if hasattr(X_test_tfidf_sample, 'toarray'):
    X_dense = X_test_tfidf_sample.toarray()
else:
    X_dense = np.array(X_test_tfidf_sample)

# shap_values may be a list of arrays (one per class) or a single 3D array
if isinstance(shap_values, list):
    shap_per_class = shap_values  # already a list
else:
    shap_per_class = [shap_values[:, :, i] for i in range(shap_values.shape[2])]

for i, class_name in enumerate(le_sent.classes_):
    sv = shap_per_class[i]
    # ensure dense
    if hasattr(sv, 'toarray'):
        sv = sv.toarray()
    sv = np.array(sv)

    shap.summary_plot(
        sv,
        X_dense,
        feature_names=feature_names,
        max_display=15,
        show=False,
        plot_type='bar'
    )
    plt.title(f'SHAP Feature Importance, {class_name}')
    plt.tight_layout()
    plt.savefig(f'outputs/shap_summary_{class_name}.png', bbox_inches='tight')
    plt.show()
    plt.close()

--- CELL 147 (markdown) ---
## Section 6, Results Interpretation

### 6.1 Error Analysis

The Logistic Regression model misclassifies 449 out of 2,410 test reviews (18.6%),
with errors distributed very unevenly across classes.

**Neutral class is the dominant failure mode.** With a 92.0% error rate, the neutral
class is virtually unpredictable for classical models. Of the 338 neutral test reviews,
163 are predicted as negative and 148 as positive, the model splits them almost
equally between the two polar classes, confirming that 3-star reviews contain genuinely
mixed language with no consistent signal distinguishing them from mild positive or
negative reviews. This is not a model failure, it reflects the inherent ambiguity of
the neutral class itself.

**Negative and positive classes are well-learned.** With error rates of only 4.7% and
8.8% respectively, the model reliably identifies clear sentiment in both directions.
The 66 positive-to-negative confusions and 42 negative-to-positive confusions represent
genuinely ambiguous reviews, the sample misclassifications confirm this: "je n'ai
jamais eu de litige avec direct assurance" is predicted negative despite being positive
(the word "litige" triggers a negative association despite the negation), and "cette
mutuelle est chère mais le traitement est" is predicted positive despite being negative
(the mixed structure confuses the bag-of-words model).

**Review length does not explain errors.** The length histogram shows that correct and
misclassified reviews follow nearly identical length distributions, peaking at 20-50
tokens and tapering off beyond 200. This confirms that misclassification is driven by
content ambiguity rather than review length, short reviews are not systematically
harder to classify than long ones.

**Word cloud of misclassified reviews** is dominated by stopwords and generic
insurance vocabulary ("de", "le", "un", "et", "pour", "assurance", "contrat") rather
than sentiment-bearing terms. This confirms the core finding: misclassified reviews use
general language without strong positive or negative markers, placing them in the
ambiguous middle ground that all models struggle to resolve.

---

### 6.2 Sentiment Detection Deep Dive

**Per-insurer sentiment breakdown** reveals clear differentiation between insurers
that aligns with our EDA findings:

- **APRIL Moto** shows the highest proportion of positive sentiment (~80%) with the
  smallest negative segment, consistent with its 3.95/5 average rating and 72.8%
  positive reviews identified in the EDA.
- **L'Olivier Assurance** shows approximately 75% positive, consistent with its 71.0%
  positive rate and 3.83/5 average rating.
- **MACIF and MAAF** show overwhelmingly negative sentiment (~90%), identifying them
  as the most problematic insurers in the corpus by predicted sentiment, neither
  appeared in our top-5 analysis but their profiles warrant attention.
- **GMF and Néoliane Santé** both show approximately 40-45% negative sentiment,
  consistent with their sub-3.0 average ratings identified in the EDA.
- **Santiane** is a notable finding, despite appearing in the top 10 by volume, it
  shows approximately 25% negative and 75% positive, suggesting a generally satisfied
  customer base that was not examined in the per-insurer summary section.

The neutral segment (orange) is consistently thin across all insurers, reflecting the
model's known inability to predict neutral reviews, not an absence of neutral
customers, but a limitation of the classifier.

**Sentiment volume over time** reveals the most analytically interesting temporal
finding in the dataset. Volume is relatively stable from 2016 to 2020, with negative
reviews consistently outnumbering positive ones throughout this period, confirming
that dissatisfied customers were more likely to write reviews during the early years
of the dataset. A dramatic structural break occurs in early 2021, with positive review
volume spiking to over 1,500 in a single month while negative volume simultaneously
spikes to approximately 650. This coincides with the accelerated digitisation of
insurance services during the COVID-19 pandemic period, newly digitalised customers
who successfully subscribed online drove the positive spike, while customers
experiencing claim delays during lockdown drove the simultaneous negative spike. Both
return to baseline by late 2021. The neutral class remains flat throughout,
consistently near zero, which reflects both the model's prediction bias and the
genuine rarity of 3-star reviews in this corpus.

---

### 6.3 Sentiment and Topic Cross-Analysis

The cross-tabulation of LDA topics with predicted sentiment produces the most
nuanced analytical finding in the project.

**Left panel, Sentiment proportion by topic.** All eight topics show approximately
50-55% negative sentiment, which at first appears uniform. However the variation
in positive proportion is significant:

- **Satisfaction Générale** (Topic 2) has the highest positive proportion (~45%),
  as expected, this topic was identified in Section 3 as the formulaic positive
  review cluster.
- **Sinistres Auto** and **Service Client Négatif** both show approximately 45-50%
  negative sentiment, confirming that claims and customer service interactions
  generate the most complaint-heavy reviews.
- **Accueil Téléphonique Positif** (Topic 7) shows a surprisingly high negative
  proportion (~50%), which at first seems contradictory. This reflects LDA topic
  assignment, reviews are assigned to their dominant topic even when they mix positive
  and negative content. A review praising telephone service in one sentence and
  complaining about price in another may be assigned to Topic 7 while still being
  classified as negative overall.

**Right panel, Average star rating by topic.** All topics cluster between 2.4 and
3.0 stars, well below the neutral threshold of 3.0, which confirms a corpus-wide
negative bias in the training reviews. This is consistent with the J-shaped
distribution identified in the EDA: the corpus is dominated by 1-star reviews, pulling
the topic averages down regardless of topic content. **Mutuelle Santé** has the lowest
average rating (approximately 2.4 stars), identifying health insurance mutual customers
as the most dissatisfied segment in the corpus. **Prévoyance & Vie** has the highest
topic average (approximately 2.95 stars), still below the neutral threshold but
suggesting that life insurance customers express slightly less extreme dissatisfaction
than auto or health insurance customers.

---

### 6.4 GloVe+BiLSTM Learned Embeddings

**Insurance terms t-SNE (Image 5)** shows meaningful semantic clustering despite the
low GloVe coverage (76.2%) on translated English reviews:

- **price** and **premium** cluster together in the lower right, correctly identified
  as synonymous financial terms in insurance contexts.
- **good**, **excellent**, **satisfied**, and **recommend** form a loose positive
  sentiment cluster in the upper centre, consistent with their co-occurrence in
  positive reviews.
- **problem** and **bad** appear in the upper right, correctly separated from the
  positive cluster.
- **service**, **customer**, and **support** cluster on the left, the customer
  service dimension identified throughout the analysis.
- **claim**, **refund**, and **payment** cluster in the centre-right, correctly
  grouped as financial transaction terms.
- **contract**, **coverage**, and **insurance** appear in the lower half, representing
  the product/contract vocabulary.
- **fast** appears isolated in the upper left, reflecting that speed/efficiency
  vocabulary co-occurs with both positive and negative reviews (fast claim resolution
  = positive; fast price increase = negative) and therefore occupies an intermediate
  embedding position.

**Top 300 words t-SNE (Image 6)** shows a diffuse cloud with limited visible
clustering, which is expected given the frozen GloVe weights and the English
translation quality limitations. Notable visible tokens include "disappointed" in the
lower left (correctly separated from positive vocabulary), "reimburse" and "contracts"
in the upper region (financial/administrative vocabulary), "damage" and "april" in
the lower right (claims and insurer name), and "<OOV>" clustered with "olivier",
confirming that insurer names were not in the GloVe vocabulary and were treated as
out-of-vocabulary tokens, which partially explains the model's lower performance
compared to the French Word2Vec model.

---

### 6.5 Model Comparison and Final Recommendations

**Three-class evaluation** reveals a clear performance hierarchy:

**CamemBERT** achieves the highest overall performance (F1-macro 0.908) but only on
binary positive/negative classification, it produces zero neutral predictions by
design. This is the recommended model for binary deployment.

**Ridge Regression** achieves the best three-class F1-macro (0.678) and is the only
model with meaningful neutral detection (Neutral F1 = 0.33, Recall = 0.36). By
reframing sentiment as a continuous star rating prediction problem, it learns a
spectrum that accommodates ambiguous 3-star reviews better than any direct
classification approach. This is the recommended model when neutral detection
is required.

**Classical TF-IDF models** (Logistic Regression, LinearSVM, Naive Bayes) cluster
between F1-macro 0.63 and 0.65, providing strong binary performance (accuracy
93.3% in binary evaluation) but near-zero neutral detection. Their key advantage is
training speed and interpretability.

**Deep learning models** (BasicEmbedding+Dense, GloVe+BiLSTM) do not outperform
classical models on this corpus, which is a significant finding. The corpus size
(~19,000 training samples after filtering) is insufficient to provide a meaningful
advantage for sequential models over TF-IDF+linear classifiers. The GloVe+BiLSTM
underperformance relative to the Basic Embedding model also confirms that the 76.2%
English GloVe coverage introduces more noise than the pre-trained weights provide
benefit for this specific translated corpus.

**Qwen2.5-1.5B zero-shot** performs at near-random levels (F1-macro 0.31), confirming
that zero-shot inference with a 1.5B parameter model is not viable for this task
without fine-tuning. The binary evaluation accuracy of 50%, equivalent to random
guessing, underscores this conclusion.

**Binary evaluation** produces a counterintuitive finding: classical TF-IDF models
(Logistic Regression at 93.3%) outperform CamemBERT (90.9%) on raw accuracy when
neutral reviews are excluded. However CamemBERT's F1-macro of 0.908 versus Logistic
Regression's 0.627 reveals why accuracy is misleading here, CamemBERT achieves
balanced precision and recall across both classes, while Logistic Regression achieves
high accuracy by exploiting the class distribution rather than learning balanced
decision boundaries.

**Conclusion:** For a production Streamlit deployment, the recommended architecture is
a two-model ensemble, CamemBERT for confident binary positive/negative predictions,
with Ridge Regression as a fallback for reviews where CamemBERT confidence is low,
providing neutral detection for genuinely ambiguous cases. This hybrid approach
combines the strengths of both models while acknowledging the fundamental difficulty
of neutral class prediction in this corpus.

### 6.6 Model Explainability, LIME and SHAP

#### LIME Local Explanations

LIME explains individual predictions by perturbing the input text and observing how 
the model's output changes, identifying which specific tokens most influenced each 
decision.

**Negative review, MAIF tendon rupture claim (52% negative confidence)**

The model correctly predicts negative but with only moderate confidence. LIME 
identifies "aucune" (0.13) and "vie" (0.09) as the two strongest positive evidence 
tokens for the negative class, "aucune prise en charge" (no coverage) and "accident 
de la vie" (life accident insurance product) are the exact phrases that anchor this 
review as a complaint about denied coverage. "Prêt", "rupture", and "compris" 
reinforce this through their co-occurrence in contract termination language. 
Counterintuitively, "maif", "sans", "bonne", "bien", and "charge" push against the 
negative prediction, the model has learned that MAIF reviews can be positive, and 
"bonne" and "bien" are positive sentiment markers that create genuine ambiguity. This 
is a good example of a legitimately hard case: a customer who was loyal for years, 
describes a specific legitimate claim, but frames it with mixed vocabulary that 
includes positive terms ("bonne", "bien", "tous les contrats possibles").

**Neutral review, pricing complaint misclassified as negative (55% negative, 
17% neutral)**

This is the most analytically revealing LIME result. The review "tout simplement trop 
chère le double du prix proposé par direct assurance" is labelled neutral (3 stars) 
but the model predicts negative with 55% probability. LIME correctly identifies why: 
"trop" (0.07) and "chère" (0.05) are strong negative price complaint signals, and 
"tout simplement" is a rhetorical intensifier. The review reads as a price complaint 
which the model, correctly from a linguistic standpoint, maps to negative sentiment. 
The 3-star rating likely reflects the customer acknowledging the insurance is adequate 
but overpriced, a nuance that is invisible to any bag-of-words model. This single 
example encapsulates the fundamental neutral class problem identified throughout the 
modelling section: the linguistic signal genuinely points negative, and only the 
rating label says otherwise.

**Positive review, telephone service praise (96% positive confidence)**

The model predicts positive with near-perfect confidence. LIME shows "très" (0.10), 
"rapide" (0.05), and "merci" (0.05) as the dominant positive features, exactly the 
top positive unigrams identified in the Section 1 EDA n-gram analysis, providing a 
direct validation that the TF-IDF model learned the right features. The word 
"interrogation" pushes against the positive prediction (−0.03), which is a 
recognisable stopword leakage artefact: "interrogation" in French can have a 
questioning/uncertain connotation that the model associates with hesitancy even though 
here it appears in "répondu à toutes mes interrogations" (answered all my questions), 
a clearly positive phrase. This confirms the limitation of unigram features for French 
negation and complex multi-word expressions.

---

#### SHAP Global Feature Importance

SHAP measures the average magnitude of each feature's contribution across 100 test 
samples, providing a global picture of which tokens drive each class's predictions.

**Negative class SHAP**

"Satisfait" (0.147) and "très" (0.101) are paradoxically the two most important 
features for the negative class. This does not mean they predict negative reviews, 
it means their absence in a review is a strong negative signal. SHAP measures 
absolute impact regardless of direction: when "satisfait" and "très" are present, 
they push strongly toward positive; when absent from a review, the model interprets 
this absence as evidence for negative. "Fuir" (0.052) and "mois" (0.044) are the 
genuinely directional negative features, "fuir" (flee/avoid) is a near-perfect 
negative classifier identified in the Section 1 trigram analysis, and "mois" captures 
the temporal grievance pattern. "Aucune", "alors", and "plus" at the lower end of the 
chart reflect the complaint vocabulary identified throughout the project.

**Neutral class SHAP**

The neutral SHAP chart is the most interesting of the three. "Très" (0.035) dominates, 
but its impact for neutral is lower in absolute magnitude than for positive or negative 
(0.135 and 0.101 respectively), confirming that neutral reviews are characterised 
precisely by the absence of strong positive or negative markers rather than the 
presence of distinctive neutral vocabulary. "Nadège" (0.029) appearing as the second 
most important feature is an artefact, it is almost certainly a reviewer name that 
appeared in a small number of 3-star reviews in the 100-sample test set, 
demonstrating the noise sensitivity of SHAP estimates on small samples. "Fuir", 
"jamais", "espère", and "voir" appearing alongside "satisfait" and "recommande" 
confirm that neutral reviews mix positive and negative vocabulary in roughly equal 
measure, which is precisely why the model cannot reliably identify them.

**Positive class SHAP**

The positive SHAP chart is the cleanest and most interpretable of the three. "Très" 
(0.135) and "satisfait" (0.119) are by far the most important features, with a sharp 
drop-off to "rapide" (0.062), "recommande" (0.060), and "prix" (0.059). This 
five-word vocabulary, très, satisfait, rapide, recommande, prix, captures the 
formulaic positive review pattern identified in the n-gram analysis in Section 1. The 
concentration of positive signal in so few features explains why positive reviews are 
reliably classified (8.8% error rate) while negative reviews are harder (4.7% error 
rate only because the class is so large). Counterintuitively, "mois", "impossible", 
"fuir", and "aucune" all appear in the positive SHAP chart, again as absence-driven 
features whose non-occurrence signals positive sentiment.

---

#### Cross-method Validation

LIME and SHAP together provide complementary validation of the EDA findings from 
Section 1. The tokens LIME identifies as locally decisive, "aucune", "fuir", "trop", 
"très", "rapide", "merci", are exactly the unigrams identified as the clearest 
sentiment separators in the stratified frequency analysis. The SHAP global importance 
charts confirm that the TF-IDF+Logistic Regression model has learned a linguistically 
coherent feature representation rather than spurious correlations. The main explainability 
finding is that the model's errors are not random, they are concentrated precisely in 
the cases where the linguistic signal and the rating label are genuinely inconsistent, 
as demonstrated by the neutral review LIME example where "trop chère" correctly 
predicts negative despite a 3-star label.

