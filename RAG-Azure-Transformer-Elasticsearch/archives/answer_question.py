import os
import urllib3
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import requests

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuration
ES_URL = "https://elastic:Z25ft0VLU2fpiqOXRSEc@localhost:9200"
ES_INDEX = "docs_rag"
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "llama3.1:8b"

# Initialisation
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
es_client = Elasticsearch(ES_URL, verify_certs=False, ssl_show_warn=False)


def ask(question: str):
    """Pose une question et obtient une réponse"""
    
    # 1. Générer l'embedding de la question
    print(f"\nQuestion: {question}")
    print("Recherche dans la base...")
    
    question_embedding = embedding_model.encode([question])[0].tolist()
    
    # 2. Rechercher dans Elasticsearch
    search_query = {
        "knn": {
            "field": "embedding",
            "query_vector": question_embedding,
            "k": 3,
            "num_candidates": 100
        },
        "_source": ["content", "metadata"]
    }
    
    results = es_client.search(index=ES_INDEX, body=search_query)
    
    # 3. Construire le contexte
    context = "\n\n".join([
        hit['_source']['content']
        for hit in results['hits']['hits']
    ])
    
    print(f"✓ {len(results['hits']['hits'])} documents trouvés")
    
    # 4. Générer la réponse avec Ollama
    print("Génération de la réponse...")
    
    prompt = f"""Documents:
{context}

Question: {question}

Réponds en te basant uniquement sur les documents ci-dessus. Si la réponse n'est pas dans les documents, dis-le.

Réponse:"""
    
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )
    
    answer = response.json()['response']
    
    # 5. Afficher la réponse
    print("\n" + "="*60)
    print("RÉPONSE:")
    print("="*60)
    print(answer)
    print("="*60)
    
    # 6. Afficher les sources
    print("\nSOURCES:")
    for i, hit in enumerate(results['hits']['hits']):
        metadata = hit['_source']['metadata']
        print(f"{i+1}. {metadata['source']} (chunk {metadata['chunk_index']})")


if __name__ == "__main__":
    # Exemple d'utilisation
    ask("What is NeRF?")