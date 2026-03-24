import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import torch
import shap

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_qwen_pipeline():
    """Returns the Qwen model for text generation / RAG."""
    logging.info("Loading Qwen2.5-1.5B-Instruct...")
    try:
        from transformers import pipeline
        return pipeline(
            "text-generation", 
            model="Qwen/Qwen2.5-1.5B-Instruct",
            device_map="auto",
            model_kwargs={"torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32}
        )
    except Exception as e:
        logging.error(f"Failed to load Qwen: {e}")
        return None

def optimal_insurer_summary(insurer_name, product_line, df, top_n_sentiments=3, max_reviews_per_sentiment=2, llm_pipeline=None):
    """
    Optimal LLM Summary Pipeline from notebook:
    Rating Stats + Theme Extraction (TF-IDF) + Weighted Stratified Sampling + LLM Synthesis.
    """
    import math
    
    # 1. Filter dataset and calculate basic stats
    if insurer_name != "All":
        df_sub = df[df['assureur'] == insurer_name].copy()
    else:
        df_sub = df.copy()
        
    if product_line != "All":
        df_sub = df_sub[df_sub['produit'] == product_line].copy()
        
    if df_sub.empty:
        return "No data available."
        
    avg_rating = df_sub['note'].mean()
    total_reviews = len(df_sub)
    
    # Stratify into broad sentiment bands (1-5 scale)
    dist = []
    for s in range(1, 6):
        count = len(df_sub[df_sub['note'] == s])
        dist.append(f"{s}⭐:{count}({count/total_reviews*100:.1f}%)")

    # 2. Key Theme Extraction using TF-IDF (per rating)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    sampled_reviews = []
    
    for rating in [1, 3, 5]: # Negative, Neutral, Positive focus layers
        df_rating = df_sub[df_sub['note'] == rating]
        if df_rating.empty: continue
            
        texts = df_rating['avis'].dropna().tolist()
        if not texts: continue
            
        tfidf = TfidfVectorizer(max_features=1000, stop_words=None, ngram_range=(1, 2))
        try:
            tfidf_matrix = tfidf.fit_transform(texts)
            centroid = tfidf_matrix.mean(axis=0)
            
            # Find representative reviews closest to centroid
            similarities = cosine_similarity(tfidf_matrix, np.asarray(centroid))
            top_indices = similarities.flatten().argsort()[::-1][:max_reviews_per_sentiment]
            
            for idx in top_indices:
                sampled_reviews.append(f"Rating {rating}/5: {texts[idx]}")
        except ValueError:
            pass # Ignore if vocabulary too small
            
    # Combine selected reviews
    selected_texts = "\n---\n".join(sampled_reviews)
    
    # 3. LLM Synthesis
    if llm_pipeline:
        prompt = f"""<|im_start|>system
Tu es un analyste expert du secteur de l'assurance. Rédige une synthèse narrative des retours clients pour l'assureur {insurer_name} (Produit: {product_line}). 
La note moyenne globale est de {avg_rating:.1f}/5 sur {total_reviews} avis.
La distribution des notes est: {', '.join(dist)}.
Structure ta réponse en 3 paragraphes:
1. Aperçu général et points forts
2. Thématiques récurrentes et points de friction
3. Recommandation finale
Concentre-toi sur les thèmes récurrents présents dans les exemples fournis. Sois objectif et professionnel. Utilise moins de 250 mots.
<|im_end|>
<|im_start|>user
Voici un échantillon structuré et pondéré des avis représentatifs:
{selected_texts}
<|im_end|>
<|im_start|>assistant
"""
        try:
            outputs = llm_pipeline(prompt, max_new_tokens=400, do_sample=True, temperature=0.3)
            return outputs[0]["generated_text"].split("<|im_start|>assistant\n")[-1].strip()
        except Exception as e:
            return f"Summary failed: {e}"
    else:
        return f"Pipeline LLM absent. Average: {avg_rating:.2f}/5, Total: {total_reviews}. Samples:\n{selected_texts}"

def explain_with_shap(model, tfidf_vec, text, label_index=None):
    """Explain single text prediction using SHAP."""
    import shap
    X_vec = tfidf_vec.transform([text])
    
    # We will compute feature importances locally using logistic regression weights 
    # to avoid needing the full training background data for shap.LinearExplainer
    
    # We will compute feature importances locally if LinearExplainer isn't easily initialized
    coef = model.coef_
    if len(coef.shape) > 1:
        if label_index is None: label_index = 0
        weights = coef[label_index] # Target class
    else:
        weights = coef
        
    features = tfidf_vec.get_feature_names_out()
    nonzero = X_vec.nonzero()[1]
    
    contributions = {features[idx]: float(X_vec[0, idx] * weights[idx]) for idx in nonzero}
    return pd.DataFrame(list(contributions.items()), columns=['word', 'impact']).sort_values('impact', key=abs, ascending=False)
    
class FAISSSearchEngine:
    """FAISS Semantic Search using paraphrase-multilingual-MiniLM-L12-v2."""
    def __init__(self, df):
        self.df = df
        self.sentences = df['avis_corrected_clean'].dropna().tolist()
        
        logging.info("Initializing FAISS with Multilingual SBERT...")
        self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.corpus_embeddings = self.embedder.encode(self.sentences, convert_to_tensor=False, normalize_embeddings=True)
        
        self.index = faiss.IndexFlatIP(self.corpus_embeddings.shape[1])
        self.index.add(self.corpus_embeddings.astype('float32'))
        
        # We need a proper map to map the filtered texts back to the original rows
        self.doc_map = df.dropna(subset=['avis_corrected_clean']).copy().reset_index(drop=True)

    def search(self, query, insurer=None, top_k=10):
        """Semantic Search."""
        query_embedding = self.embedder.encode([query], normalize_embeddings=True).astype('float32')
        scores, semantic_indices = self.index.search(query_embedding, 50)  # Over-fetch for filtering
        
        candidates = self.doc_map.iloc[semantic_indices[0]].copy()
        candidates['similarity_score'] = scores[0]
        
        if insurer and insurer != "All":
            candidates = candidates[candidates['assureur'] == insurer]
            
        return candidates.head(top_k)

def local_rag_query(query, search_engine, llm_pipeline=None, top_k=10, insurer=None):
    """RAG Strategy using Qwen2.5 and FAISS."""
    retrieved_docs = search_engine.search(query, insurer=insurer, top_k=top_k)
    
    if retrieved_docs.empty:
        return "Je n'ai pas trouvé d'avis correspondant à votre recherche.", retrieved_docs
        
    context = ""
    for i, (_, row) in enumerate(retrieved_docs.iterrows()):
        rating = row['note'] if pd.notna(row['note']) else "N/A"
        company = row['assureur'] if pd.notna(row['assureur']) else "Inconnu"
        context += f"AVIS {i+1} [Assureur: {company} | Note: {rating}⭐]:\n{row['avis']}\n\n"
        
    prompt = f"""<|im_start|>system
Tu es un analyste en assurance. Réponds à la question de l'utilisateur de manière précise en te basant UNIQUEMENT sur les avis clients fournis dans le contexte.
Règles:
- Cite les noms des assureurs et les notes associés.
- Sois objectif et structure ta réponse.
<|im_end|>
<|im_start|>user
QUESTION: {query}

CONTEXTE:
{context}
<|im_end|>
<|im_start|>assistant
"""
    if llm_pipeline:
        try:
            result = llm_pipeline(prompt, max_new_tokens=400, do_sample=True, temperature=0.1, return_full_text=False)
            response = result[0]['generated_text'].split("<|im_start|>assistant\n")[-1].strip()
        except Exception as e:
            response = f"Erreur de génération : {e}"
    else:
        response = "Modèle LLM non chargé. Lisez les avis ci-dessous."
        
    return response, retrieved_docs
