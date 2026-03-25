import streamlit as st
import pandas as pd
import os
import sys
import logging
from PIL import Image

# CRITICAL: Disable GPU to prevent XLA/PTX compilation crashes (Error 134)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

logging.basicConfig(level=logging.INFO)

# Add code dir to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(BASE_DIR, 'code')
if CODE_DIR not in sys.path: sys.path.append(CODE_DIR)

import preprocessing, unsupervised, supervised, analysis

st.set_page_config(
    page_title="Insurance Reviews NLP Insights - Final Project", 
    page_icon="🛡️", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
.stApp { background-color: #ffffff; color: #31333f; }
[data-testid="stSidebar"] { background-color: #f8fafc; }
.stMetric { background-color: #f1f5f9; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; }
.main-header { font-size: 2.5rem; font-weight: 800; color: #1e293b; margin-bottom: 2rem; }
.section-header { font-size: 1.5rem; font-weight: 700; color: #334155; margin-top: 2rem; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_dataset():
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    
    # Try CSV first
    csv_path = os.path.join(OUTPUT_DIR, "reviews_clean.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if 'note' in df.columns:
            df['note'] = pd.to_numeric(df['note'], errors='coerce').fillna(0).astype(int)
        if 'assureur' in df.columns:
            df['assureur'] = df['assureur'].astype(str).str.strip()
        if 'avis_corrected_clean' not in df.columns and 'avis_clean' in df.columns:
            df['avis_corrected_clean'] = df['avis_clean']
        if 'note' in df.columns and 'sentiment' not in df.columns:
            df['sentiment'] = df['note'].apply(lambda n: 'positive' if n >= 4 else ('negative' if n <= 2 else 'neutral'))
        return df

    for filename in ["insurance_reviews_cleaned.parquet", "reviews_corrected_and_translated.parquet", "reviews_corrected.parquet"]:
        path = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(path):
            df = pd.read_parquet(path)
            if 'note' in df.columns:
                df['note'] = pd.to_numeric(df['note'], errors='coerce').fillna(0).astype(int)
            if 'assureur' in df.columns:
                df['assureur'] = df['assureur'].astype(str).str.strip()
            if 'avis_corrected_clean' not in df.columns and 'avis_clean' in df.columns:
                df['avis_corrected_clean'] = df['avis_clean']
            if 'note' in df.columns and 'sentiment' not in df.columns:
                df['sentiment'] = df['note'].apply(lambda n: 'positive' if n >= 4 else ('negative' if n <= 2 else 'neutral'))
            return df
    return pd.DataFrame()

@st.cache_resource
def get_ml_models():
    df = load_dataset()
    resources = {}
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Search Engines (with Persistence)
    try:
        idx_path = os.path.join(OUTPUT_DIR, "faiss_index.bin")
        emb_path = os.path.join(OUTPUT_DIR, "corpus_embeddings.npy")
        bm25_path = os.path.join(OUTPUT_DIR, "bm25_index.pkl")
        resources["search_semantic"] = analysis.FAISSSearchEngine(df, index_path=idx_path, embs_path=emb_path)
        resources["search_keyword"] = analysis.BM25SearchEngine(df, cache_path=bm25_path)
    except Exception as e:
        logging.error(f"Search Engine init failed: {e}")
        
    # 2. RAG Generator (Qwen 2.5)
    resources["rag_llm"] = analysis.get_qwen_pipeline()
    
    # 3. Supervised Models
    try:
        from sklearn.model_selection import train_test_split
        train_pool = df.dropna(subset=['note', 'avis_corrected_clean']).copy()
        train_df = train_pool.sample(min(len(train_pool), 15000), random_state=42)
        X = train_df['avis_corrected_clean'].astype(str).values
        y_sent = train_df['sentiment'].values
        y_note = train_df['note'].values
        X_train, X_test, y_train_sent, y_test_sent, y_train_stars, y_test_stars = train_test_split(
            X, y_sent, y_note, test_size=0.2, random_state=42, stratify=y_sent
        )
        resources["ridge"], resources["tfidf"], resources["mae"] = supervised.train_ridge_regression(X_train, X_test, y_train_stars, y_test_stars)
        resources["lr_model"], _ = supervised.train_logistic_regression(X_train, X_test, y_train_sent, y_test_sent)

        # 4. Neural Model (Bi-LSTM with persistence)
        nn_path, tok_path, le_path = [os.path.join(OUTPUT_DIR, f) for f in ["nn_bilstm.keras", "nn_tokenizer.pkl", "nn_labelencoder.pkl"]]
        if all(os.path.exists(p) for p in [nn_path, tok_path, le_path]):
            import pickle
            try:
                resources["nn_model"] = tf.keras.models.load_model(nn_path)
                with open(tok_path, 'rb') as f: resources["nn_tokenizer"] = pickle.load(f)
                with open(le_path, 'rb') as f: resources["le"] = pickle.load(f)
            except Exception as e:
                logging.warning(f"Model load failed ({e}). Purging cache for retraining...")
                for p in [nn_path, tok_path, le_path]:
                    if os.path.exists(p): os.remove(p)
                # Fallthrough to retraining logic
        
        if not resources.get("nn_model"): # This condition handles both initial absence and failed loading
            keras_pool = train_pool.dropna(subset=['sentiment'])
            keras_df = keras_pool.groupby('sentiment').apply(lambda x: x.sample(min(len(x), 1500))).reset_index(drop=True)
            from tensorflow.keras.utils import to_categorical
            from sklearn.preprocessing import LabelEncoder
            import pickle
            le = LabelEncoder(); y_sent_enc = le.fit_transform(keras_df['sentiment'].values)
            y_cat = to_categorical(y_sent_enc)
            from tensorflow.keras.preprocessing.text import Tokenizer
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
            tokenizer.fit_on_texts(keras_df['avis_corrected_clean'].astype(str))
            X_seq = pad_sequences(tokenizer.texts_to_sequences(keras_df['avis_corrected_clean'].astype(str)), maxlen=150)
            X_tr, X_va, y_tr, y_va = train_test_split(X_seq, y_cat, test_size=0.15, random_state=42, stratify=y_cat)
            resources["nn_model"] = supervised.train_bilstm(X_tr, X_va, y_tr, y_va, vocab_size=10000, max_len=150)
            resources["nn_model"].save(nn_path)
            with open(tok_path, 'wb') as f: pickle.dump(tokenizer, f)
            with open(le_path, 'wb') as f: pickle.dump(le, f)
            resources["nn_tokenizer"], resources["le"] = tokenizer, le
    except Exception as e:
        logging.error(f"Modeling failed: {e}")
    return resources

def calculate_distances(w1, w2, w2v_model):
    """Calculates Cosine and Euclidean distances between two words."""
    import numpy as np
    try:
        v1 = w2v_model.wv[w1]
        v2 = w2v_model.wv[w2]
        cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        euc_dist = np.linalg.norm(v1 - v2)
        return float(cos_sim), float(euc_dist)
    except Exception:
        return None, None

def show_image_if_exists(filename, caption=None):
    path = os.path.join(BASE_DIR, "outputs", filename)
    if os.path.exists(path):
        # Using use_container_width=True as it's still widely supported, or width='stretch' if on latest
        st.image(Image.open(path), caption=caption, use_container_width=True)

def main():
    st.sidebar.markdown("# 🛡️ NLP Project Lab")
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Project Lifecycle", [
        "📊 1. Exploration & Data Quality",
        "🔍 2. Unsupervised Discovery",
        "⚖️ 3. Supervised Intelligence",
        "💬 4. Hybrid Search & RAG"
    ])
    
    if st.sidebar.button("🔄 Reload Model Data"):
        st.cache_data.clear(); st.cache_resource.clear()
        
    df = load_dataset()
    if df.empty:
        st.error("No data found in outputs/. Please run the notebook first."); return
    resources = get_ml_models()

    if page == "📊 1. Exploration & Data Quality":
        st.markdown('<div class="main-header">📊 Data Exploration & Cleansing</div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3); m1.metric("Total Reviews", f"{len(df):,}"); m2.metric("Avg Rating", f"{df['note'].mean():.2f}/5"); m3.metric("Insurers", df['assureur'].nunique())
        
        tab1, tab2, tab3 = st.tabs(["Distributions & Trends", "Lexical Quality", "Data Preview & Cleaning Audit"])
        with tab1:
            col1, col2 = st.columns(2)
            with col1: 
                show_image_if_exists("rating_distribution.png", "Stars Distribution")
                st.write("### Top Insurers by Volume")
                st.dataframe(df['assureur'].value_counts().reset_index().rename(columns={'assureur':'Insurer', 'count':'Reviews'}).head(10))
            with col2: 
                show_image_if_exists("sentiment_over_time.png", "Review Volume Over Time")
                st.write("### Reviews by Product")
                if 'produit' in df.columns:
                    st.dataframe(df['produit'].value_counts().reset_index().rename(columns={'produit':'Product', 'count':'Reviews'}).head(10))
            show_image_if_exists("polarity_vs_rating.png", "Polarity vs Rating Correlation")
            
        with tab2:
            col1, col2 = st.columns(2)
            with col1: show_image_if_exists("spelling_corrections_languagetool.png", "LanguageTool Cleaning Impact")
            with col2: show_image_if_exists("text_length_french.png", "Review Length Analysis")
            show_image_if_exists("ngrams_by_rating_avis_clean.png", "N-Grams Comparison (1-star vs 5-star)")
        
        with tab3:
            st.subheader("Raw vs Cleaned & Translated Metadata")
            st.write("Audit the results of spaCy lemmatization and spelling correction below.")
            cols_to_show = ['assureur', 'note', 'avis', 'avis_corrected_clean']
            existing_cols = [c for c in cols_to_show if c in df.columns]
            st.dataframe(df[existing_cols].head(20))
            st.info("The cleaning pipeline removes HTML tags, punctuation, and applies LanguageTool suggest corrections for better vector representation.")

    elif page == "🔍 2. Unsupervised Discovery":
        st.markdown('<div class="main-header">🔍 Unsupervised Insights</div>', unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["Topic Modeling (LDA)", "Anomaly Detection", "Embedding Lab"])
        
        with tab1:
            st.subheader("Topic Modeling (Latent Dirichlet Allocation)")
            col1, col2 = st.columns(2)
            with col1: show_image_if_exists("lda_topic_distribution.png", "Topic Frequency")
            with col2: show_image_if_exists("sentiment_theme_analysis.png", "Sentiment per Topic")
            lda_path = os.path.join(BASE_DIR, "outputs", "lda_vis.html")
            if os.path.exists(lda_path): st.subheader("Intertopic Distance Map (pyLDAvis)"); st.components.v1.html(open(lda_path, 'r', encoding='utf-8').read(), height=800, scrolling=True)

        with tab2:
            st.subheader("🕵️ Outlier Detection (Isolation Forest)")
            st.info("Identifies suspicious or generic reviews based on TF-IDF variance.")
            if st.button("Scan for Anomalies"):
                df_anom, _ = unsupervised.detect_anomalies(df)
                anomalies = df_anom[df_anom['is_anomaly'] == -1]
                st.write(f"Detected **{len(anomalies)}** outliers. Example suspect reviews:")
                st.dataframe(anomalies[['assureur', 'note', 'avis_corrected_clean', 'anomaly_score']].head(15))

        with tab3:
            st.subheader("🔠 Embedding Vector Lab")
            col1, col2 = st.columns([1, 1.2])
            with col1: show_image_if_exists("embeddings_visualization.png", "PCA Word Clusters")
            with col2:
                st.write("### Distance Validator")
                w1 = st.text_input("Word 1", "remboursement"); w2 = st.text_input("Word 2", "attente")
                if st.button("Calculate Similarity"):
                    w2v = unsupervised.train_word2vec(df['avis_corrected_clean'].dropna().tolist())
                    sim, dist = calculate_distances(w1, w2, w2v)
                    if sim: st.success(f"Cosine Similarity: **{sim:.4f}**"); st.info(f"Euclidean Distance: **{dist:.4f}**")

    elif page == "⚖️ 3. Supervised Intelligence":
        st.markdown('<div class="main-header">⚖️ Supervised Learning & Insights</div>', unsafe_allow_html=True)
        
        # 1. Benchmarks
        st.subheader("📊 Performance Benchmarks")
        col1, col2 = st.columns(2)
        with col1: show_image_if_exists("model_comparison_final.png", "Binary Leaderboard")
        with col2: show_image_if_exists("three_class_comparison.png", "Multi-Class Performance (Neutral class challenge)")
        
        st.markdown("---")
        
        # 2. Unified Predictor
        st.subheader("🎯 Integrated Sentiment Predictor")
        st.write("Compare Classical ML, Advanced Deep Learning, and Transformer architectures in real-time.")
        user_input = st.text_area("Review to Analyze", "Service client fantastique, un grand merci à l'équipe !")
        
        if st.button("🚀 Analyze Sentiment"):
            if not user_input:
                st.warning("Please enter some text.")
            else:
                # shared prep
                cleaned = preprocessing.clean_text(user_input)
                
                # 3. Model Outputs (Side-by-Side Comparison)
                st.markdown("#### Model Predictions")
                c1, c2, c3 = st.columns(3)
                
                # Ridge (Classical)
                if resources["ridge"]:
                    vec = resources["tfidf"]
                    X_in = vec.transform([cleaned])
                    stars = resources["ridge"].predict(X_in)[0]
                    cat = supervised.stars_to_sentiment(stars)
                    c1.markdown(f"**Ridge Baseline** (Stars)")
                    if stars >= 3.5: c1.success(f"{stars:.2f} ⭐ ({cat.upper()})")
                    elif stars >= 2.5: c1.warning(f"{stars:.2f} ⭐ ({cat.upper()})")
                    else: c1.error(f"{stars:.2f} ⭐ ({cat.upper()})")
                
                # Bi-LSTM (Custom Neural)
                if resources.get("nn_model"):
                    from tensorflow.keras.preprocessing.sequence import pad_sequences
                    seq = pad_sequences(resources["nn_tokenizer"].texts_to_sequences([cleaned]), maxlen=150)
                    pred = resources["nn_model"].predict(seq, verbose=0)
                    idx = pred.argmax(axis=1)[0]
                    cat_nn = resources["le"].classes_[idx]
                    conf = pred[0][idx]
                    c2.markdown(f"**Bi-LSTM Neural** (Custom)")
                    if "positive" in cat_nn: c2.success(f"{cat_nn.upper()} ({conf*100:.1f}%)")
                    elif "neutral" in cat_nn: c2.warning(f"{cat_nn.upper()} ({conf*100:.1f}%)")
                    else: c2.error(f"{cat_nn.upper()} ({conf*100:.1f}%)")
                
                # CamemBERT (State-of-the-Art)
                try:
                    res_bert = supervised.run_camembert_inference([user_input])
                    label_bert = res_bert[0]
                    c3.markdown(f"**CamemBERT** (Transformer)")
                    if "pos" in label_bert.lower(): c3.success(label_bert.upper())
                    else: c3.error(label_bert.upper())
                except:
                    c3.info("Transformer inference skipped.")
                
                # 3. SHAP Explanation directly below
                st.markdown("---")
                st.subheader("🔍 Local Token Influence (Why the models predicted this)")
                try:
                    with st.spinner("Calculating word weights..."):
                        shap_df = analysis.explain_with_shap(resources["ridge"], resources["tfidf"], cleaned)
                        st.dataframe(shap_df.head(15), use_container_width=True)
                except Exception as e:
                    st.error(f"SHAP failed: {e}")

        # 4. Neural Curves
        st.markdown("---")
        st.subheader("📉 Advanced DL - Training Diagnostics")
        col_c1, col_c2 = st.columns(2)
        with col_c1: show_image_if_exists("training_glove_bilstm.png", "Bi-LSTM Accuracy/Loss Curves")
        with col_c2: show_image_if_exists("training_basic_embed.png", "Basic Embed Accuracy/Loss Curves")

    elif page == "💬 4. Hybrid Search & RAG":
        st.markdown('<div class="main-header">💬 Search, Retrieval & Synthesis</div>', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["Smart Search (RAG)", "Insurer Synthesis"])
        
        with tab1:
            q = st.text_input("Posez une question sur le marché (ex: 'Qui a les meilleurs remboursements ?')")
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1: ins = st.selectbox("Filter", ["All"] + sorted(list(df['assureur'].unique())))
            with c2: stype = st.radio("Method", ["Semantic (FAISS)", "Keyword (BM25)"])
            with c3: k = st.slider("Docs", 1, 20, 5)
            if q:
                eng = resources["search_semantic"] if "Semantic" in stype else resources["search_keyword"]
                if eng:
                    docs = eng.search(q, insurer=None if ins=="All" else ins, top_k=k)
                    st.subheader(f"📚 Top Reviews ({stype})")
                    for _, r in docs.iterrows():
                        with st.expander(f"⭐ {r['note']} - {r['assureur']} (Score: {r['similarity_score']:.3f})"):
                            st.write(f"**Review:** {r['avis']}")
                    st.markdown("---"); st.subheader("🤖 Qwen 2.5 LLM Answer")
                    with st.spinner("Synthesizing answer..."):
                        if resources["rag_llm"]: st.write(analysis.generate_rag_response(q, docs, resources["rag_llm"]))
        
        with tab2:
            st.subheader("Executive Insurer Profiles")
            target = st.selectbox("Select Insurer", sorted(df['assureur'].unique()))
            
            c1, c2 = st.columns([1, 1.2])
            
            with c1:
                st.markdown(f"#### 📊 Rating Distribution for {target}")
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                ins_df = df[df['assureur'].astype(str) == str(target).strip()]
                if not ins_df.empty:
                    fig, ax = plt.subplots(figsize=(8, 6)) # bit taller for columns
                    palette = ["#4D79D1", "#F18B52", "#66CC6F", "#D15D5D", "#9476B8"]
                    sns.countplot(x='note', data=ins_df, hue='note', palette=palette, order=[1, 2, 3, 4, 5], ax=ax, legend=False)
                    ax.set_title(f"Score Frequency ({target})")
                    for p in ax.patches:
                        h = p.get_height()
                        if h > 0: ax.annotate(f'{int(h)}', (p.get_x() + p.get_width() / 2., h), ha='center', va='bottom')
                    st.pyplot(fig)
                else: st.warning("No data found.")

            with c2:
                st.markdown(f"#### 🤖 AI Executive Summary")
                if st.button(f"Generate Summary for {target}"):
                    with st.spinner("Analyzing reviews..."):
                        sumry = analysis.optimal_insurer_summary(target, "All", df, llm_pipeline=resources["rag_llm"])
                        st.info(sumry)
                else:
                    st.write("Click the button above to synthesize recent reviews into a strategic summary.")

            st.markdown("---")
            st.markdown("#### ⚖️ Market Context (Full Group Comparison)")
            show_image_if_exists("sentiment_by_insurer.png", f"Market-wide Sentiment Distribution ({target} highlighted)")
            show_image_if_exists("heatmap_insurer_product.png", "Detailed Insurer/Product Cross-Analysis")

if __name__ == "__main__":
    main()
