"""
RAG System - Question/Réponse avec Azure OpenAI et Elasticsearch

Workflow:
1. Question posée par l'utilisateur
2. Génération de l'embedding de la question
3. Recherche des documents les plus proches dans Elasticsearch
4. Construction du prompt avec les documents
5. Envoi au modèle Azure OpenAI
6. Retour de la réponse avec sources
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
import urllib3
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI

# Désactiver les warnings SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Charger les variables d'environnement depuis .env
load_dotenv()

# Créer l'application FastAPI
app = FastAPI(title="RAG System - Azure OpenAI")

# Configuration CORS pour permettre les appels depuis un frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# CONFIGURATION - Chargement des variables d'environnement
# ============================================================================

# Elasticsearch
ES_URL = os.getenv("ES_URL", "https://elastic:Z25ft0VLU2fpiqOXRSEc@localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "docs_rag")

# Azure OpenAI
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

# ============================================================================
# INITIALISATION - Au démarrage de l'application
# ============================================================================

print(" Démarrage du système RAG...")

# 1. Charger le modèle d'embeddings (pour transformer la question en vecteur)
print(" Chargement du modèle d'embeddings...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print(" Modèle d'embeddings chargé (384 dimensions)")

# 2. Connexion à Elasticsearch
print(" Connexion à Elasticsearch...")
es_client = Elasticsearch(
    ES_URL,
    verify_certs=False,
    ssl_show_warn=False
)
if es_client.ping():
    print(f"Connecté à Elasticsearch - Index: {ES_INDEX}")
else:
    print("Impossible de se connecter à Elasticsearch")

# 3. Initialiser le client Azure OpenAI
azure_client = None
if AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY:
    try:
        azure_client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION
        )
        print(f"Azure OpenAI initialisé - Déploiement: {AZURE_OPENAI_DEPLOYMENT}")
    except Exception as e:
        print(f"Erreur Azure OpenAI: {e}")
else:
    print(" Azure OpenAI non configuré (vérifiez .env)")

print("Système RAG prêt!\n")

# ============================================================================
# MODÈLES DE DONNÉES (pour l'API)
# ============================================================================

class QueryRequest(BaseModel):
    """Requête de l'utilisateur"""
    question: str
    top_k: Optional[int] = 3  # Nombre de documents à récupérer
    include_sources: Optional[bool] = True  # Inclure les sources dans la réponse
    temperature: Optional[float] = 0.7  # Créativité du modèle (0-1)
    max_tokens: Optional[int] = 1000  # Longueur max de la réponse


class Source(BaseModel):
    """Un document source"""
    content: str  # Contenu du document
    metadata: dict  # Métadonnées (fichier, page, etc.)
    score: float  # Score de similarité


class QueryResponse(BaseModel):
    """Réponse du système"""
    answer: str  # Réponse générée
    sources: Optional[List[Source]] = None  # Documents utilisés
    model_used: str  # Modèle utilisé


# ============================================================================
# FONCTION 1 : RECHERCHE DES DOCUMENTS DANS ELASTICSEARCH
# ============================================================================

def search_similar_chunks(question: str, top_k: int = 3):
    """
    Recherche les documents les plus similaires à la question dans Elasticsearch
    
    Étapes:
    1. Transforme la question en vecteur (embedding)
    2. Recherche vectorielle dans Elasticsearch (KNN)
    3. Retourne les top_k documents les plus proches
    
    Args:
        question: La question de l'utilisateur
        top_k: Nombre de documents à retourner (défaut: 3)
    
    Returns:
        Liste de dictionnaires contenant content, metadata, score
    """
    
    print(f" Recherche pour: '{question}'")
    
    # ÉTAPE 1: Générer l'embedding de la question
    # Transforme le texte en vecteur de 384 dimensions
    question_embedding = embedding_model.encode([question])[0].tolist()
    print(f"   → Embedding généré: {len(question_embedding)} dimensions")
    
    # ÉTAPE 2: Créer la requête de recherche vectorielle
    search_query = {
        "knn": {  # K-Nearest Neighbors (recherche des plus proches voisins)
            "field": "embedding",  # Champ contenant les vecteurs
            "query_vector": question_embedding,  # Vecteur de la question
            "k": top_k,  # Nombre de résultats
            "num_candidates": 100  # Nombre de candidats à évaluer
        },
        "_source": ["content", "metadata"]  # Champs à retourner
    }
    
    # ÉTAPE 3: Exécuter la recherche dans Elasticsearch
    results = es_client.search(index=ES_INDEX, body=search_query)
    
    # ÉTAPE 4: Extraire les documents trouvés
    chunks = []
    for hit in results['hits']['hits']:
        chunks.append({
            "content": hit['_source']['content'],
            "metadata": hit['_source']['metadata'],
            "score": hit['_score']  # Score de similarité
        })
    
    print(f"   → {len(chunks)} documents trouvés")
    
    return chunks


# ============================================================================
# FONCTION 2 : GÉNÉRATION DE LA RÉPONSE AVEC AZURE OPENAI
# ============================================================================

def generate_answer_azure(question: str, chunks: list, temperature: float = 0.7, max_tokens: int = 1000):
    """
    Génère une réponse avec Azure OpenAI en utilisant les documents trouvés
    
    Étapes:
    1. Construit le contexte avec les documents trouvés
    2. Crée le prompt avec instructions + contexte + question
    3. Envoie au modèle Azure OpenAI
    4. Retourne la réponse générée
    
    Args:
        question: La question de l'utilisateur
        chunks: Liste des documents trouvés
        temperature: Créativité (0=factuel, 1=créatif)
        max_tokens: Longueur max de la réponse
    
    Returns:
        Réponse générée par le modèle
    """
    
    # Vérifier que Azure OpenAI est configuré
    if not azure_client:
        raise HTTPException(
            status_code=500, 
            detail="Azure OpenAI non configuré. Vérifiez votre fichier .env"
        )
    
    print(f"🤖 Génération de la réponse avec {AZURE_OPENAI_DEPLOYMENT}...")
    
    # ÉTAPE 1: Construire le contexte avec les documents
    # Format: [Document 1]\nContenu...\n\n[Document 2]\nContenu...
    context = "\n\n".join([
        f"[Document {i+1}]\n{chunk['content']}"
        for i, chunk in enumerate(chunks)
    ])
    
    print(f"   → Contexte: {len(context)} caractères")
    
    # ÉTAPE 2: Créer le prompt complet
    # Le prompt contient: instructions + contexte + question
    prompt = f"""Documents de référence:
{context}

Question: {question}

Instructions:
- Réponds UNIQUEMENT en te basant sur les documents fournis ci-dessus
- Si la réponse n'est pas dans les documents, dis-le clairement
- Cite les sources (Document 1, Document 2, etc.) quand c'est pertinent
- Sois concis et précis

Réponse:"""
    
    try:
        # ÉTAPE 3: Envoyer au modèle Azure OpenAI
        response = azure_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {
                    "role": "system",
                    "content": "Tu es un assistant intelligent qui répond aux questions en te basant uniquement sur les documents fournis."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # ÉTAPE 4: Extraire la réponse
        answer = response.choices[0].message.content
        print(f"   → Réponse générée: {len(answer)} caractères")
        
        return answer
    
    except Exception as e:
        print(f"Erreur: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur Azure OpenAI: {str(e)}")


# ============================================================================
# ENDPOINTS DE L'API
# ============================================================================

@app.get("/")
async def root():
    """Page d'accueil de l'API"""
    return {
        "message": "RAG System - Elasticsearch, Azure OpenAI",
        "version": "1.0.0",
        "status": {
            "elasticsearch": "connected" if es_client.ping() else "disconnected",
            "azure_openai": "configured" if azure_client else "not_configured"
        }
    }


@app.get("/health")
async def health():
    """Vérification de la santé du système"""
    
    status = {
        "api": "ok",
        "elasticsearch": "ok" if es_client.ping() else "error",
        "azure_openai": "ok" if azure_client else "not_configured"
    }
    
    # Compter les documents indexés
    if status["elasticsearch"] == "ok":
        try:
            count = es_client.count(index=ES_INDEX)
            status["documents_indexed"] = count['count']
        except:
            status["documents_indexed"] = 0
    
    return status


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    ENDPOINT PRINCIPAL - Répond à une question
    
    Workflow complet:
    1. Reçoit la question
    2. Recherche documents similaires dans Elasticsearch
    3. Construit le prompt avec les documents
    4. Génère la réponse avec Azure OpenAI
    5. Retourne réponse + sources
    """
    
    print("\n" + "="*60)
    print(f"Nouvelle question: {request.question}")
    print("="*60)
    
    try:
        # ÉTAPE 1: Rechercher les documents pertinents
        chunks = search_similar_chunks(request.question, request.top_k)
        
        if not chunks:
            return QueryResponse(
                answer="Aucun document pertinent trouvé dans la base de données.",
                sources=[],
                model_used=AZURE_OPENAI_DEPLOYMENT
            )
        
        # ÉTAPE 2: Générer la réponse avec Azure OpenAI
        answer = generate_answer_azure(
            request.question, 
            chunks,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        # ÉTAPE 3: Préparer les sources si demandé
        sources = None
        if request.include_sources:
            sources = [
                Source(
                    content=chunk['content'],
                    metadata=chunk['metadata'],
                    score=chunk['score']
                )
                for chunk in chunks
            ]
        
        print("✅ Réponse prête!")
        print("="*60 + "\n")
        
        # ÉTAPE 4: Retourner la réponse complète
        return QueryResponse(
            answer=answer, 
            sources=sources,
            model_used=AZURE_OPENAI_DEPLOYMENT
        )
    
    except Exception as e:
        print(f"❌ Erreur: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Statistiques sur les documents indexés"""
    
    try:
        count = es_client.count(index=ES_INDEX)
        
        # Récupérer la liste des fichiers sources
        search_result = es_client.search(
            index=ES_INDEX,
            body={
                "size": 0,
                "aggs": {
                    "sources": {
                        "terms": {
                            "field": "metadata.source",
                            "size": 100
                        }
                    }
                }
            }
        )
        
        sources = []
        if 'aggregations' in search_result:
            for bucket in search_result['aggregations']['sources']['buckets']:
                sources.append({
                    "filename": bucket['key'],
                    "chunks": bucket['doc_count']
                })
        
        return {
            "total_chunks": count['count'],
            "index": ES_INDEX,
            "sources": sources,
            "model": AZURE_OPENAI_DEPLOYMENT
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# LANCEMENT DE L'APPLICATION
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n🌐 Démarrage du serveur FastAPI...")
    print("📍 URL: http://localhost:8000")
    print("📚 Documentation: http://localhost:8000/docs")
    print("\nAppuyez sur Ctrl+C pour arrêter\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)