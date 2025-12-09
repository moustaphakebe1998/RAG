"""
Point d'entrée principal de l'API RAG
Utilise tous les services pour créer l'API FastAPI
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

#  IMPORTS RELATIFS
from .models import QueryRequest, QueryResponse, HealthResponse, StatsResponse
from .rag_service import RAGService
from .config import Config

# ============================================================================
app = FastAPI(
    title=Config.API_TITLE,
    version=Config.API_VERSION,
    description="Système de Question/Réponse avec Elasticsearch et Azure OpenAI"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialiser le service RAG (au démarrage de l'application)
try:
    Config.validate()  # Valider la configuration
    rag_service = RAGService()
except Exception as e:
    print(f" Erreur lors de l'initialisation: {e}")
    raise


# ============================================================================
# ENDPOINTS DE L'API
# ============================================================================

@app.get("/")
async def root():
    """Page d'accueil de l'API"""
    return {
        "message": Config.API_TITLE,
        "version": Config.API_VERSION,
        "status": rag_service.get_health()
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Vérification de la santé du système"""
    return rag_service.get_health()


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Endpoint principal - Répond à une question
    
    Workflow:
    1. Reçoit la question
    2. Recherche documents similaires dans Elasticsearch
    3. Génère la réponse avec Azure OpenAI
    4. Retourne réponse + sources
    """
    
    try:
        answer, sources = rag_service.query(
            question=request.question,
            top_k=request.top_k,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            include_sources=request.include_sources
        )
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            model_used=Config.AZURE_OPENAI_DEPLOYMENT
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Statistiques sur les documents indexés"""
    try:
        return rag_service.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# LANCEMENT DE L'APPLICATION
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n Démarrage du serveur FastAPI...")
    print(f" URL: http://{Config.API_HOST}:{Config.API_PORT}")
    print(f" Documentation: http://{Config.API_HOST}:{Config.API_PORT}/docs")
    
    uvicorn.run(
        app, 
        host=Config.API_HOST, 
        port=Config.API_PORT
    )