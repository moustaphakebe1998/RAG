"""
Configuration du système RAG
Charge les variables d'environnement et centralise tous les paramètres
"""

import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()


class Config:
    """Configuration centralisée de l'application"""
    
    # ========================================================================
    # ELASTICSEARCH
    # ========================================================================
    ES_URL = os.getenv("ES_URL", "https://elastic:Z25ft0VLU2fpiqOXRSEc@localhost:9200")
    ES_INDEX = os.getenv("ES_INDEX", "docs_rag")
    
    # ========================================================================
    # AZURE OPENAI
    # ========================================================================
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-11-01")
    
    # ========================================================================
    # MODÈLE D'EMBEDDINGS
    # ========================================================================
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION = 384
    
    # ========================================================================
    # PARAMÈTRES RAG
    # ========================================================================
    DEFAULT_TOP_K = 3  # Nombre de documents à récupérer
    DEFAULT_TEMPERATURE = 0.7  # Créativité du modèle
    DEFAULT_MAX_TOKENS = 1000  # Longueur max de la réponse
    
    # ========================================================================
    # API
    # ========================================================================
    API_TITLE = "RAG System - Azure OpenAI"
    API_VERSION = "1.0.0"
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    
    @classmethod
    def validate(cls):
        """Valide que les configurations critiques sont présentes"""
        errors = []
        
        if not cls.AZURE_OPENAI_ENDPOINT:
            errors.append("AZURE_OPENAI_ENDPOINT manquant dans .env")
        
        if not cls.AZURE_OPENAI_API_KEY:
            errors.append("AZURE_OPENAI_API_KEY manquant dans .env")
        
        if errors:
            raise ValueError("Erreurs de configuration:\n" + "\n".join(f"- {e}" for e in errors))
        
        return True