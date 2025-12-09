import os
import urllib3
from datetime import datetime
from typing import List
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
import uuid

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# Configuration
ES_URL = os.getenv("ES_URL", "https://elastic:Z25ft0VLU2fpiqOXRSEc@localhost:9200")
ES_INDEX = "docs_rag"
DATA_PATH = "./files"

print("Chargement du modèle d'embeddings...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Modèle chargé avec succès!")

es_client = Elasticsearch(
    ES_URL,
    verify_certs=False,
    ssl_show_warn=False
)


def test_es_connection():
    if es_client.ping():
        print("Connexion réussie à Elasticsearch !")
        return True
    else:
        print("Impossible de se connecter à Elasticsearch.")
        return False


def create_index():
    """Crée l'index Elasticsearch avec le mapping approprié"""
    if es_client.indices.exists(index=ES_INDEX):
        print(f"L'index '{ES_INDEX}' existe déjà")
        response = input("Voulez-vous le supprimer et le recréer? (y/n): ")
        if response.lower() == 'y':
            es_client.indices.delete(index=ES_INDEX)
            print(f"Index '{ES_INDEX}' supprimé")
        else:
            print("Utilisation de l'index existant")
            return
    
    mapping = {
        "mappings": {
            "properties": {
                "content": {
                    "type": "text",
                    "analyzer": "standard"
                },
                "embedding": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": True,
                    "similarity": "cosine"
                },
                "metadata": {
                    "properties": {
                        "source": {"type": "keyword"},
                        "page": {"type": "integer"},
                        "chunk_index": {"type": "integer"},
                        "total_chunks": {"type": "integer"},
                        "upload_date": {"type": "date"}
                    }
                }
            }
        },
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        }
    }
    
    es_client.indices.create(index=ES_INDEX, body=mapping)
    print(f"Index '{ES_INDEX}' créé avec succès")


def load_documents(path: str):
    """Charge les documents depuis un répertoire"""
    print(f"\nChargement des documents depuis: {path}")
    
    if not os.path.exists(path):
        print(f"Le répertoire '{path}' n'existe pas")
        return []
    
    loader = DirectoryLoader(path, glob="*.txt")
    documents = loader.load()
    
    print(f"{len(documents)} documents chargés")
    
    return documents


def split_documents(documents):
    """Découpe les documents en chunks"""
    print("\nDécoupage des documents en chunks...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=200,
        separators=["\n", "\n\n", r"(?<=[.!?])", "", " "]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    print(f"{len(chunks)} chunks créés")
    
    if chunks:
        chunk_lengths = [len(chunk.page_content) for chunk in chunks]
        avg_length = sum(chunk_lengths) / len(chunk_lengths)
        print(f"Taille moyenne: {avg_length:.0f} caractères")
        print(f"Min: {min(chunk_lengths)}, Max: {max(chunk_lengths)}")
    
    return chunks


def generate_embeddings(chunks):
    """Génère les embeddings pour les chunks"""
    print("\nGénération des embeddings...")
    
    texts = [chunk.page_content for chunk in chunks]
    
    embeddings = embedding_model.encode(
        texts, 
        show_progress_bar=True,
        batch_size=32
    )
    
    print(f"{len(embeddings)} embeddings générés")
    
    return embeddings.tolist()


def store_in_elasticsearch(chunks, embeddings):
    """Stocke les chunks et leurs embeddings dans Elasticsearch"""
    print("\nStockage dans Elasticsearch...")
    
    actions = []
    upload_date = datetime.now().isoformat()
    total_chunks = len(chunks)
    
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        metadata = chunk.metadata
        
        doc = {
            "_index": ES_INDEX,
            "_id": str(uuid.uuid4()),
            "_source": {
                "content": chunk.page_content,
                "embedding": embedding,
                "metadata": {
                    "source": metadata.get("source", "unknown"),
                    "page": metadata.get("page", 0),
                    "chunk_index": i,
                    "total_chunks": total_chunks,
                    "upload_date": upload_date
                }
            }
        }
        actions.append(doc)
    
    success, failed = bulk(es_client, actions, refresh=True)
    
    print(f"{success} documents indexés avec succès")
    if failed:
        print(f"{len(failed)} documents ont échoué")
    
    return success


def get_index_stats():
    """Affiche les statistiques de l'index"""
    try:
        if not es_client.indices.exists(index=ES_INDEX):
            print(f"L'index '{ES_INDEX}' n'existe pas")
            return
        
        stats = es_client.count(index=ES_INDEX)
        print(f"\nStatistiques de l'index '{ES_INDEX}':")
        print(f"Nombre total de documents: {stats['count']}")
        
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
        
        if 'aggregations' in search_result:
            sources = search_result['aggregations']['sources']['buckets']
            if sources:
                print(f"\nFichiers indexés:")
                for source in sources:
                    filename = os.path.basename(source['key'])
                    print(f"- {filename}: {source['doc_count']} chunks")
    
    except Exception as e:
        print(f"Erreur lors de la récupération des stats: {str(e)}")


def process_directory(path: str = DATA_PATH):
    """Pipeline complet de traitement"""
    print("\n" + "="*60)
    print(f"TRAITEMENT DU RÉPERTOIRE: {path}")
    print("="*60)
    
    try:
        documents = load_documents(path)
        
        if not documents:
            print("Aucun document à traiter")
            return
        
        chunks = split_documents(documents)
        
        if not chunks:
            print("Aucun chunk créé")
            return
        
        embeddings = generate_embeddings(chunks)
        
        indexed_count = store_in_elasticsearch(chunks, embeddings)
        
        print(f"\n{'='*60}")
        print(f"SUCCÈS: {indexed_count} documents indexés dans '{ES_INDEX}'")
        print(f"{'='*60}\n")
        
        get_index_stats()
        
    except Exception as e:
        print(f"\nERREUR lors du traitement: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Fonction principale - Exécution automatique"""
    print("\n" + "="*60)
    print("SCRIPT DE STOCKAGE ELASTICSEARCH")
    print("="*60 + "\n")
    
    # Test de connexion
    print("Test de connexion à Elasticsearch...")
    if not test_es_connection():
        return
    
    print("\n" + "-"*60)
    
    # Créer l'index
    print("\nConfiguration de l'index...")
    create_index()
    
    print("\n" + "-"*60)
    
    # Traiter automatiquement le répertoire par défaut
    process_directory(DATA_PATH)


if __name__ == "__main__":
    main()


#https://localhost:9200/docs_rag/_search?pretty&_source=content,embedding,metadata