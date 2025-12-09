"""
Service Azure OpenAI
Gère l'interaction avec Azure OpenAI pour la génération de réponses
"""

from openai import AzureOpenAI
from typing import List, Dict
from fastapi import HTTPException

# IMPORT RELATIF (avec le point)
from .config import Config


class AzureOpenAIService:
    """Service pour interagir avec Azure OpenAI"""
    
    def __init__(self):
        """Initialise le client Azure OpenAI"""
        
        print("Initialisation du service Azure OpenAI...")
        
        # Vérifier que les configurations sont présentes
        if not Config.AZURE_OPENAI_ENDPOINT or not Config.AZURE_OPENAI_API_KEY:
            raise ValueError("Configuration Azure OpenAI manquante dans .env")
        
        # Initialiser le client
        try:
            self.client = AzureOpenAI(
                azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
                api_key=Config.AZURE_OPENAI_API_KEY,
                api_version=Config.AZURE_OPENAI_API_VERSION
            )
            print(f" Azure OpenAI initialisé - Déploiement: {Config.AZURE_OPENAI_DEPLOYMENT}")
        except Exception as e:
            raise ConnectionError(f"Erreur lors de l'initialisation Azure OpenAI: {e}")
    
    def generate_answer(
        self, 
        question: str, 
        chunks: List[Dict], 
        temperature: float = 0.7, 
        max_tokens: int = 1000
    ) -> str:
        """Génère une réponse en utilisant Azure OpenAI"""
        
        print(f" Génération de la réponse avec {Config.AZURE_OPENAI_DEPLOYMENT}...")
        
        # 1. Construire le contexte avec les documents
        context = self._build_context(chunks)
        print(f"   → Contexte: {len(context)} caractères")
        
        # 2. Créer le prompt
        prompt = self._build_prompt(question, context)
        
        # 3. Appeler Azure OpenAI
        try:
            response = self.client.chat.completions.create(
                model=Config.AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # 4. Extraire la réponse
            answer = response.choices[0].message.content
            print(f"   → Réponse générée: {len(answer)} caractères")
            
            return answer
        
        except Exception as e:
            print(f"    Erreur: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"Erreur Azure OpenAI: {str(e)}"
            )
    
    def _build_context(self, chunks: List[Dict]) -> str:
        """Construit le contexte à partir des chunks"""
        return "\n\n".join([
            f"[Document {i+1}]\n{chunk['content']}"
            for i, chunk in enumerate(chunks)
        ])
    
    def _build_prompt(self, question: str, context: str) -> str:
        """Construit le prompt complet"""
        return f"""Documents de référence:
{context}

Question: {question}

Instructions:
- Réponds UNIQUEMENT en te basant sur les documents fournis ci-dessus
- Si la réponse n'est pas dans les documents, dis-le clairement
- Cite les sources (Document 1, Document 2, etc.) quand c'est pertinent
- Sois concis et précis

Réponse:"""
    
    def _get_system_prompt(self) -> str:
        """Retourne le prompt système"""
        return "Tu es un assistant intelligent qui répond aux questions en te basant uniquement sur les documents fournis."
    
    def is_configured(self) -> bool:
        """Vérifie si le service est correctement configuré"""
        return self.client is not None