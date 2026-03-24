import streamlit as st
import pandas as pd
import os
import sys
import logging
from PIL import Image

logging.basicConfig(level=logging.INFO)

# Add code dir to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(BASE_DIR, 'code')
if CODE_DIR not in sys.path: sys.path.append(CODE_DIR)

import preprocessing, unsupervised, supervised, analysis

st.set_page_config(page_title="Insurance NLP Insights", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp { background-color: #ffffff; color: #31333f; }
[data-testid="stSidebar"] { background-color: #f8fafc; }
.stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border: 1px solid #e2e8f0; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_dataset():
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    
    # Try CSV first if the user explicitly mentioned it
    csv_path = os.path.join(OUTPUT_DIR, "reviews_clean.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        logging.info(f"Loaded schema from {csv_path}")
        if 'avis_corrected_clean' not in df.columns and 'avis_clean' in df.columns:
            df['avis_corrected_clean'] = df['avis_clean']
        if 'note' in df.columns and 'sentiment' not in df.columns:
            df['sentiment'] = df['note'].apply(lambda n: 'positive' if n >= 4 else ('negative' if n <= 2 else 'neutral'))
        return df

    for filename in ["insurance_reviews_cleaned.parquet", "reviews_corrected_and_translated.parquet", "reviews_corrected.parquet"]:
        path = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(path):
            df = pd.read_parquet(path)
            logging.info(f"Loaded schema from {path}")
            # Map for analysis component if needed
            if 'avis_corrected_clean' not in df.columns and 'avis_clean' in df.columns:
                df['avis_corrected_clean'] = df['avis_clean']
            if 'note' in df.columns and 'sentiment' not in df.columns:
                df['sentiment'] = df['note'].apply(lambda n: 'positive' if n >= 4 else ('negative' if n <= 2 else 'neutral'))
            return df
    
    # Fallback
    DATA_DIR = os.path.join(BASE_DIR, "data")
    if os.path.exists(DATA_DIR):
        raw_df = preprocessing.load_all_data(DATA_DIR)
        df = preprocessing.run_full_pipeline(raw_df.head(10000)) # Small batch for fallback
        df['sentiment'] = df['note'].apply(lambda n: 'positive' if n >= 4 else ('negative' if n <= 2 else 'neutral'))
        return df
    return pd.DataFrame()

@st.cache_resource
def get_ml_models():
    df = load_dataset()
    resources = {}
    
    # 1. Search Engine
    try:
        resources["search"] = analysis.FAISSSearchEngine(df)
    except Exception as e:
        logging.error(f"Search Engine init failed: {e}")
        resources["search"] = None
        
    # 2. RAG Generator (Qwen 2.5)
    resources["rag_llm"] = analysis.get_qwen_pipeline()
    
    # 3. Supervised Models
    try:
        train_pool = df[df['type'] == 'train'].copy() if 'type' in df.columns else df.copy()
        train_pool = train_pool.dropna(subset=['note', 'avis_corrected_clean'])
        train_df = train_pool.sample(min(len(train_pool), 15000), random_state=42)
        
        X_train, X_test, y_train, y_test = supervised.prepare_modeling_data(train_df, label_col='sentiment')
        _, _, y_train_stars, y_test_stars = supervised.prepare_modeling_data(train_df, label_col='note')
        
        # Ridge Regression for Stars (Best 3-class)
        ridge_model, tfidf_vec, mae = supervised.train_ridge_regression(X_train, X_test, y_train_stars, y_test_stars)
        resources["ridge"] = ridge_model
        resources["tfidf"] = tfidf_vec
        resources["mae"] = mae
        
        # Logistic Regression for Explainability
        lr_model, _ = supervised.train_logistic_regression(X_train, X_test, y_train, y_test)
        resources["lr_model"] = lr_model

    except Exception as e:
        logging.error(f"Supervised modeling failed: {e}")
        resources["ridge"] = None
        
    return resources

def show_image_if_exists(filename, caption=None):
    path = os.path.join(BASE_DIR, "outputs", filename)
    if os.path.exists(path):
        st.image(Image.open(path), caption=caption, use_container_width=True)

def main():
    st.sidebar.title("🛡️ NLP Analysis Hub")
    if st.sidebar.button("🔄 Clear App Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        
    page = st.sidebar.selectbox("Navigation", ["📊 Market Intelligence", "🏢 Insurer Deep-Dive", "🔎 Smart Retrieval (RAG)", "🎯 Prediction & Explainability"])
    
    df = load_dataset()
    if df.empty:
        st.error("No data found.")
        return
        
    resources = get_ml_models()

    if page == "📊 Market Intelligence":
        st.title("📊 Market Intelligence (EDA & NLP)")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Reviews", f"{len(df):,}")
        m2.metric("Market Avg Rating", f"{df['note'].mean():.2f} / 5")
        m3.metric("Primary Insurers", df['assureur'].nunique())
        
        tab1, tab2, tab3 = st.tabs(["Distributions & Trends", "Word Clouds & N-Grams", "Topic Modeling (LDA)"])
        
        with tab1:
            st.subheader("Ratings & Sentiment Over Time")
            col1, col2 = st.columns(2)
            with col1: show_image_if_exists("rating_distribution.png", "Corpus-wide Rating Distribution")
            with col2: show_image_if_exists("polarity_vs_rating.png", "Text Polarity vs Star Rating")
            show_image_if_exists("sentiment_over_time.png", "Predicted Sentiment Volume Over Time")
            show_image_if_exists("text_length_french.png", "Review Length Analysis")
            show_image_if_exists("spelling_corrections_languagetool.png", "Spelling Corrections via LanguageTool")

        with tab2:
            st.subheader("Lexical Analysis by Sentiment")
            show_image_if_exists("wordclouds_by_rating.png", "Word Clouds by 1-Star vs 5-Star")
            show_image_if_exists("ngrams_by_rating_avis_clean.png", "Top Unigrams/Bigrams/Trigrams by Class")
            
        with tab3:
            st.subheader("Latent Dirichlet Allocation (8 Topics)")
            show_image_if_exists("lda_topic_distribution.png", "Review Count per Dominant Topic")
            show_image_if_exists("sentiment_theme_analysis.png", "Sentiment Proportion & Avg Rating by Topic")
            
            # Show pyldavis interactively
            lda_path = os.path.join(BASE_DIR, "outputs", "lda_vis.html")
            if not os.path.exists(lda_path): lda_path = os.path.join(BASE_DIR, "outputs", "lda_viz.html")
            
            if os.path.exists(lda_path):
                st.components.v1.html(open(lda_path, 'r', encoding='utf-8').read(), height=800, scrolling=True)

    elif page == "🏢 Insurer Deep-Dive":
        st.title("🏢 Insurer Detailed Analysis")
        insurer = st.selectbox("Select Target Insurer", sorted(df['assureur'].unique()))
        
        show_image_if_exists("sentiment_by_insurer.png")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader(f"Optimal LLM Summary ({insurer})")
            if st.button("Generate Executive Summary"):
                with st.spinner("Compiling stratified sample & synthesizing with Qwen2.5..."):
                    summary = analysis.optimal_insurer_summary(
                        insurer, "All", df, llm_pipeline=resources["rag_llm"]
                    )
                    st.success(summary)
        with col2:
            show_image_if_exists("heatmap_insurer_product.png", "Insurer / Product Quality Matrix")
            show_image_if_exists("rating_trends_by_product.png")

    elif page == "🔎 Smart Retrieval (RAG)":
        st.title("🔎 Semantic Search & Local RAG")
        query = st.text_input("Ask a question (e.g. 'Problème de remboursement lent')")
        
        col1, col2 = st.columns(2)
        with col1: sel_insurer = st.selectbox("Restrict to Insurer", ["All"] + sorted(list(df['assureur'].unique())))
        with col2: top_k = st.slider("Results", 1, 20, 5)

        if query:
            if resources["search"]:
                search_ins = None if sel_insurer == "All" else sel_insurer
                ans, docs = analysis.local_rag_query(query, resources["search"], resources["rag_llm"], top_k, insurer=search_ins)
                
                st.subheader("🤖 Qwen2.5 LLM Answer")
                st.write(ans)
                
                st.markdown("---")
                st.subheader("📚 Top Retrieved Reviews (FAISS SBERT)")
                for i, row in docs.iterrows():
                    with st.expander(f"⭐ {row['note']} - {row['assureur']} (Score FAISS: {row['similarity_score']:.3f})"):
                        st.write(f"**Original:** {row['avis']}")
            else:
                st.warning("Search engine unavailable.")

    elif page == "🎯 Prediction & Explainability":
        st.title("🎯 Classical ML Prediction & Explainability")
        st.markdown("Uses **Ridge Regression** for continuous stars (handles neutral well) & **Logistic Regression** for SHAP.")
        
        user_input = st.text_area("Review Text", "Le service client est catastrophique, fuyez cette assurance.", height=100)
        
        if st.button("Predict Sentiment"):
            if resources["ridge"] and user_input:
                vec = resources["tfidf"]
                X_in = vec.transform([user_input])
                
                # Predict Stars
                pred_stars = resources["ridge"].predict(X_in)[0]
                pred_sentiment = supervised.stars_to_sentiment(pred_stars)
                
                st.metric("Predicted Stars (Continuous)", f"{pred_stars:.2f} ⭐", delta=pred_sentiment, delta_color="off")
                
                # SHAP Explanation via LR
                st.subheader("🔍 Local Feature Importance (Text Coefficients)")
                try:
                    # Target the predicted class index in LR
                    lr_classes = list(resources["lr_model"].classes_)
                    label_idx = lr_classes.index(pred_sentiment) if pred_sentiment in lr_classes else 0
                    
                    shap_df = analysis.explain_with_shap(resources["lr_model"], vec, user_input, label_idx)
                    st.dataframe(shap_df.head(10))
                except Exception as e:
                    st.error(f"Explanation failed: {e}")
                    
        st.markdown("---")
        st.subheader("Model Comparison (Notebook Extract)")
        col1, col2 = st.columns(2)
        with col1: show_image_if_exists("model_comparison_final.png", "Binary Positive/Negative Metrics")
        with col2: show_image_if_exists("three_class_comparison.png", "Neutral Class Difficulties")
        
        # Binary prediction using CamemBERT
        st.subheader("HuggingFace CamemBERT (SOTA Binary)")
        if st.button("Test Binary Classifier"):
            with st.spinner("Downloading/Loading tblard/tf-allocine..."):
                try:
                    res = supervised.run_camembert_inference([user_input])
                    st.success(f"CamemBERT Predicts: **{res[0].upper()}**")
                except Exception as e:
                    st.error(f"CamemBERT inference error: {e}")

if __name__ == "__main__":
    main()
