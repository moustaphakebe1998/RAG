"""
Service Elasticsearch
Gère la connexion et les opérations sur Elasticsearch
"""

import urllib3
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from typing import List, Dict

#  IMPORT RELATIF (avec le point)
from .config import Config

# Désactiver les warnings SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class ElasticsearchService:
    """Service pour interagir avec Elasticsearch"""
    
    def __init__(self):
        """Initialise la connexion Elasticsearch et le modèle d'embeddings"""
        
        print(" Initialisation du service Elasticsearch...")
        
        # Connexion à Elasticsearch
        self.client = Elasticsearch(
            Config.ES_URL,
            verify_certs=False,
            ssl_show_warn=False
        )
        
        # Vérifier la connexion
        if self.client.ping():
            print(f" Connecté à Elasticsearch - Index: {Config.ES_INDEX}")
        else:
            raise ConnectionError("Impossible de se connecter à Elasticsearch")
        
        # Charger le modèle d'embeddings
        print(f" Chargement du modèle d'embeddings: {Config.EMBEDDING_MODEL_NAME}...")
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL_NAME)
        print(f" Modèle d'embeddings chargé ({Config.EMBEDDING_DIMENSION} dimensions)")
    
    def search_similar_chunks(self, question: str, top_k: int = 3) -> List[Dict]:
        """Recherche les chunks les plus similaires à la question"""
        
        print(f" Recherche pour: '{question}'")
        
        # 1. Générer l'embedding de la question
        question_embedding = self.embedding_model.encode([question])[0].tolist()
        print(f"   → Embedding généré: {len(question_embedding)} dimensions")
        
        # 2. Créer la requête de recherche vectorielle (KNN)
        search_query = {
            "knn": {
                "field": "embedding",
                "query_vector": question_embedding,
                "k": top_k,
                "num_candidates": 100
            },
            "_source": ["content", "metadata"]
        }
        
        # 3. Exécuter la recherche
        results = self.client.search(index=Config.ES_INDEX, body=search_query)
        
        # 4. Extraire les résultats
        chunks = []
        for hit in results['hits']['hits']:
            chunks.append({
                "content": hit['_source']['content'],
                "metadata": hit['_source']['metadata'],
                "score": hit['_score']
            })
        
        print(f"   → {len(chunks)} documents trouvés")
        
        return chunks
    
    def count_documents(self) -> int:
        """Compte le nombre total de documents dans l'index"""
        try:
            result = self.client.count(index=Config.ES_INDEX)
            return result['count']
        except:
            return 0
    
    def get_stats(self) -> Dict:
        """Récupère les statistiques de l'index"""
        
        # Compter les documents
        total_chunks = self.count_documents()
        
        # Récupérer la liste des sources
        search_result = self.client.search(
            index=Config.ES_INDEX,
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
            "total_chunks": total_chunks,
            "sources": sources
        }
    
    def is_connected(self) -> bool:
        """Vérifie si la connexion à Elasticsearch est active"""
        return self.client.ping()