"""
Modèles de données Pydantic pour l'API
Définit les structures de requêtes et réponses
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class QueryRequest(BaseModel):
    """Requête de l'utilisateur pour poser une question"""
    
    question: str = Field(..., description="La question à poser au système RAG")
    top_k: Optional[int] = Field(3, description="Nombre de documents à récupérer", ge=1, le=10)
    include_sources: Optional[bool] = Field(True, description="Inclure les sources dans la réponse")
    temperature: Optional[float] = Field(0.7, description="Créativité du modèle (0-1)", ge=0, le=1)
    max_tokens: Optional[int] = Field(1000, description="Longueur maximale de la réponse", ge=100, le=4000)
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is NeRF?",
                "top_k": 3,
                "include_sources": True,
                "temperature": 0.7,
                "max_tokens": 1000
            }
        }


class Source(BaseModel):
    """Un document source utilisé pour générer la réponse"""
    
    content: str = Field(..., description="Contenu du chunk de texte")
    metadata: dict = Field(..., description="Métadonnées du document (source, page, etc.)")
    score: float = Field(..., description="Score de similarité avec la question")


class QueryResponse(BaseModel):
    """Réponse du système RAG"""
    
    answer: str = Field(..., description="Réponse générée par le modèle")
    sources: Optional[List[Source]] = Field(None, description="Documents utilisés comme contexte")
    model_used: str = Field(..., description="Nom du modèle utilisé pour la génération")


class HealthResponse(BaseModel):
    """État de santé du système"""
    
    api: str
    elasticsearch: str
    azure_openai: str
    documents_indexed: Optional[int] = None


class StatsResponse(BaseModel):
    """Statistiques sur les documents indexés"""
    
    total_chunks: int
    index: str
    sources: List[dict]
    model: str