"""
Service RAG Principal
Orchestre Elasticsearch et Azure OpenAI pour le système RAG complet
"""

from typing import Tuple, List, Optional

# IMPORTS RELATIFS (avec le point)
from .models import Source
from .elasticsearch_service import ElasticsearchService
from .azure_service import AzureOpenAIService
from .config import Config


class RAGService:
    """Service principal qui orchestre tout le système RAG"""
    
    def __init__(self):
        """Initialise tous les services"""
        
        print(" Démarrage du système RAG...")
        
        # Initialiser Elasticsearch
        self.es_service = ElasticsearchService()
        
        # Initialiser Azure OpenAI
        self.azure_service = AzureOpenAIService()
        
        print("Système RAG prêt!\n")
    
    def query(
        self,
        question: str,
        top_k: int = Config.DEFAULT_TOP_K,
        temperature: float = Config.DEFAULT_TEMPERATURE,
        max_tokens: int = Config.DEFAULT_MAX_TOKENS,
        include_sources: bool = True
    ) -> Tuple[str, Optional[List[Source]]]:
        """Pipeline complet RAG: recherche + génération"""
        
        print("\n" + "="*60)
        print(f" Nouvelle question: {question}")
        print("="*60)
        
        # ÉTAPE 1: Rechercher les documents pertinents
        chunks = self.es_service.search_similar_chunks(question, top_k)
        
        if not chunks:
            return "Aucun document pertinent trouvé dans la base de données.", []
        
        # ÉTAPE 2: Générer la réponse avec Azure OpenAI
        answer = self.azure_service.generate_answer(
            question=question,
            chunks=chunks,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # ÉTAPE 3: Préparer les sources si demandé
        sources = None
        if include_sources:
            sources = [
                Source(
                    content=chunk['content'],
                    metadata=chunk['metadata'],
                    score=chunk['score']
                )
                for chunk in chunks
            ]
        
        print(" Réponse prête!")
        print("="*60 + "\n")
        
        return answer, sources
    
    def get_health(self) -> dict:
        """Retourne l'état de santé du système"""
        
        health = {
            "api": "ok",
            "elasticsearch": "ok" if self.es_service.is_connected() else "error",
            "azure_openai": "ok" if self.azure_service.is_configured() else "not_configured"
        }
        
        # Ajouter le nombre de documents
        if health["elasticsearch"] == "ok":
            health["documents_indexed"] = self.es_service.count_documents()
        
        return health
    
    def get_stats(self) -> dict:
        """Retourne les statistiques du système"""
        
        stats = self.es_service.get_stats()
        stats["index"] = Config.ES_INDEX
        stats["model"] = Config.AZURE_OPENAI_DEPLOYMENT
        
        return stats