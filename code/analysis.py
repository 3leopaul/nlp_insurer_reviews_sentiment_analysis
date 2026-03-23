import pandas as pd
import numpy as np
import shap
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import logging
import os
from textblob import TextBlob

# Initialize Summarization Pipeline (cached)
_summarizer = None

def summarize_reviews(texts, max_length=130, min_length=30):
    """
    Scoring Requirement: Summary by Insurer using T5/BART.
    Generates a coherent executive summary from multiple reviews.
    """
    global _summarizer
    if _summarizer is None:
        logging.info("Initializing Summarizer (sshleifer/distilbart-cnn-12-6)...")
        _summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    
    # Combine texts into a single context
    context = " ".join(texts[:10]) # Use top 10 for speed
    if len(context) < 100: return "Insufficient data for meaningful summarization."
    
    # BART has a max input token limit (usually 1024)
    summary = _summarizer(context[:2500], max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def explain_with_shap(model, X_train, X_test, vectorizer):
    """Phase 4.1: SHAP interpretation for model predictions."""
    # Transforming data to vectorized form
    X_train_vec = vectorizer.transform(X_train).toarray()
    X_test_vec = vectorizer.transform(X_test).toarray()
    
    # SHAP Explainer (KernelExplainer for generic models)
    # Using a subset for speed
    explainer = shap.KernelExplainer(model.predict_proba, X_train_vec[:50])
    shap_values = explainer.shap_values(X_test_vec[:5])
    
    logging.info("SHAP values calculated for sample test cases.")
    return explainer, shap_values

class InsuranceSearchEngine:
    """Phase 4.2: Functional IR with FAISS & BM25."""
    def __init__(self, df):
        self.df = df
        self.sentences = df['avis_cleaning_1'].tolist()
        
        # Initialize BM25
        tokenized_corpus = [doc.split() for doc in self.sentences]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Initialize FAISS (SBERT)
        self.embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.corpus_embeddings = self.embedder.encode(self.sentences, convert_to_tensor=False)
        self.index = faiss.IndexFlatL2(self.corpus_embeddings.shape[1])
        self.index.add(self.corpus_embeddings.astype('float32'))
        
        # Cross-Encoder (Initialize on demand)
        self.reranker = None

    def _get_reranker(self):
        if self.reranker is None:
            logging.info("Initializing Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)...")
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        return self.reranker

    def search(self, query, insurer=None, subject=None, top_k=10, use_reranking=True):
        """Advanced Hybrid Search with Reranking."""
        # 1. Semantic Search (Get 50 candidates)
        query_embedding = self.embedder.encode([query]).astype('float32')
        distances, semantic_indices = self.index.search(query_embedding, 50)
        
        # 2. Keyword Search (BM25)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[::-1][:50]
        
        # 3. Combine Candidates (Hybrid)
        candidate_indices = list(set(semantic_indices[0]) | set(bm25_indices))
        candidates = self.df.iloc[candidate_indices].copy()
        
        # 4. Mandatory Metadata Filtering
        if insurer and insurer != "All":
            candidates = candidates[candidates['assureur'] == insurer]
        if subject and 'subject' in candidates.columns:
            candidates = candidates[candidates['subject'] == subject]
            
        if candidates.empty: return candidates
        
        # 5. Cross-Encoder Reranking (The heavy lifting)
        if use_reranking and len(candidates) > 1:
            reranker = self._get_reranker()
            # Prepare pairs of (Query, Review)
            pairs = [[query, str(row['avis_en'])] for _, row in candidates.iterrows()]
            scores = reranker.predict(pairs)
            candidates['rerank_score'] = scores
            candidates = candidates.sort_values(by='rerank_score', ascending=False)
        
        return candidates.head(top_k)

def calculate_search_mse(results, expected_intent_score):
    """Phase 4.2: Evaluation (MSE) between intent and actual stars."""
    if results.empty:
        return 0
    actual_avg_rating = results['note'].mean()
    mse = (expected_intent_score - actual_avg_rating) ** 2
    return mse

def local_rag_query(query, search_engine, llm_pipeline=None, top_k=10):
    """Phase 4.3: High-Quality RAG Strategy."""
    # 1. Retrieve Reranked Context
    retrieved_docs = search_engine.search(query, top_k=top_k, use_reranking=True)
    
    if retrieved_docs.empty:
        return "I'm sorry, I couldn't find any reviews that match your query.", retrieved_docs
        
    # 2. Format Context (Include Metadata for attribution)
    context = ""
    for i, (_, row) in enumerate(retrieved_docs.iterrows()):
        rating = row['note'] if pd.notna(row['note']) else "N/A"
        company = row['assureur'] if pd.notna(row['assureur']) else "Unknown"
        context += f"REVIEW {i+1} [Company: {company} | Rating: {rating}⭐]:\n{row['avis_en']}\n\n"
    
    # 3. Instruction-First Prompt (Highly Structured for Qwen)
    messages = [
        {"role": "system", "content": (
            "You are a professional insurance analyst. Your task is to provide a clear, evidence-based answer to the user's question.\n"
            "STRUCTURE:\n"
            "1. DIRECT ANSWER: Start with a 1-2 sentence direct response to the question (e.g., 'The cheapest company according to these reviews is X because...').\n"
            "2. DETAILED ANALYSIS: Group findings into 'Positive Feedback' and 'Negative Feedback' or themes.\n"
            "3. CONCLUSION: A final summary of the recommendation.\n\n"
            "STRICT RULES:\n"
            "- You MUST cite the Star Rating for every claim (e.g., 'Company A (5⭐) is praised for...').\n"
            "- If the user asks for 'the best' or 'the cheapest', you MUST identify a leading candidate based on the frequency and sentiment of the reviews.\n"
            "- Only use the provided reviews. Do not use outside knowledge."
        )},
        {"role": "user", "content": f"USER QUESTION: {query}\n\nPROVIDED DATA:\n{context}"}
    ]
    
    if llm_pipeline:
        try:
            # 4. Use Tokenizer Chat Template if available
            tokenizer = llm_pipeline.tokenizer
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
            # 5. Generate with Decoder-style parameters
            # Causal LLMs need return_full_text=False to only get the new answer
            result = llm_pipeline(
                prompt, 
                max_new_tokens=256,
                do_sample=True,
                temperature=0.1, # Keep it factual
                top_p=0.9,
                return_full_text=False
            )
            response = result[0]['generated_text']
        except Exception as e:
            logging.error(f"RAG Generation failed: {e}")
            response = f"Error during generation: {e}"
    else:
        # Fallback or Mock if no pipeline provided
        response = "LLM pipeline not initialized. Showing top matches below as context."
        
    return response, retrieved_docs

def detect_rating_anomalies(df, polarity_threshold=0.4):
    """
    Notebook 'Brain': Flags reviews where text sentiment contradicts the star rating.
    High polarity + 1 star OR Low polarity + 5 star.
    """
    if 'avis_en' not in df.columns or 'note' not in df.columns:
        return df
    
    logging.info("Analyzing sentiment/rating anomalies...")
    df = df.copy()
    df['polarity'] = df['avis_en'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    
    # Flag 1: Positive text but 1-star
    df['is_anomaly'] = False
    df.loc[(df['polarity'] > polarity_threshold) & (df['note'] == 1), 'is_anomaly'] = True
    
    # Flag 2: Negative text but 5-star
    df.loc[(df['polarity'] < -0.4) & (df['note'] == 5), 'is_anomaly'] = True
    
    anomaly_count = df['is_anomaly'].sum()
    logging.info(f"Detected {anomaly_count} sentiment/rating anomalies.")
    return df

def get_temporal_stats(df):
    """
    Aggregates rating performance over time (monthly).
    Useful for Phase 5 Streamlit visualization.
    """
    if 'date_exp' not in df.columns or 'note' not in df.columns:
        return None
    
    df = df.copy()
    df['date_exp'] = pd.to_datetime(df['date_exp'], errors='coerce')
    df['month'] = df['date_exp'].dt.to_period('M')
    
    stats = df.groupby('month').agg(
        avg_rating=('note', 'mean'),
        review_count=('note', 'count')
    ).reset_index()
    
    stats['month'] = stats['month'].astype(str)
    return stats

if __name__ == "__main__":
    pass
