import os
import ssl
import urllib3
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, SpacyTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
#import lancedb
from langchain_community.vectorstores import LanceDB
from langchain_community.document_loaders import DirectoryLoader
#from PyPDF2 import PdfReader
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from langchain_community.vectorstores import ElasticVectorSearch
PATH ="docs"
#embedding = OpenAIEmbeddings()
#elasticsearch_url="http://172.17.183.127:9200"
#db = ElasticVectorSearch.from_documents(documents, embedding, elasticsearch_url=elasticsearch_url,index_name="docs")


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
def load_documents(path):
    loader = DirectoryLoader(path, glob="*.pdf")
    documents = loader.load()
    return documents

# Chargement des documents

# Configuration de l'URL Elasticsearch
elasticsearch_url = 'https://localhost:9200'  # Remplacez ceci par l'URL de votre cluster Elasticsearch

# Configuration du contexte SSL pour ignorer la vérification du certificat
ca_cert_path = '/home/kebe/langchain/elasticshearch/http_ca.crt'

# Création d'un contexte SSL personnalisé
ssl_context = ssl.create_default_context(cafile=ca_cert_path)
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# Instanciation du client Elasticsearch avec le contexte SSL personnalisé
es = Elasticsearch(
    [elasticsearch_url],
    basic_auth=('elastic', 'QANe83Fe+mfgKSOd94Zv'),  # Remplacez par vos identifiants
    ssl_context=ssl_context
)
# Vérification de la connexion
if es.ping():
    print("Connexion réussie à Elasticsearch.")
else:
    print("Impossible de se connecter à Elasticsearch. Vérifiez l'URL et les paramètres SSL.")

documents = load_documents(PATH)

# Définition du text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=200,
    separators=["\n", "\n\n", "(?<=\.!-/)","", " "]
)

# Division des documents en données
data = text_splitter.split_documents(documents)

# Préparation des actions d'indexation
index_name = 'file_chatbot'  # Remplacez 'votre_index' par le nom de l'index que vous souhaitez utiliser
actions = [
    {
        '_index': index_name,
        '_id': f'document_{idx}',
        '_source': {
            'page_content': doc.page_content,
            'metadata': doc.metadata
        }
    }
    for idx, doc in enumerate(data)
]

# Indexation par lot pour améliorer les performances
bulk(es, actions) 