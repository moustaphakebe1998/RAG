import os
import ssl
import urllib3
from langchain_openai.embeddings import OpenAIEmbeddings
#from langchain.text_splitter import CharacterTextSplitter, SpacyTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
#import lancedb
#from langchain_community.vectorstores import LanceDB
from langchain_community.document_loaders import DirectoryLoader
#from PyPDF2 import PdfReader
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from langchain_community.vectorstores import ElasticVectorSearch
from langchain_elasticsearch import ElasticsearchStore

#https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html
PATH = "docs"
# Chemin vers le certificat de l'autorité de certification
ca_cert_path = '/home/kebe/langchain/elasticshearch/http_ca.crt'

# Création d'un contexte SSL personnalisé
ssl_context = ssl.create_default_context(cafile=ca_cert_path)
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# Instanciation du client Elasticsearch avec le contexte SSL personnalisé
es = Elasticsearch(
    ['https://localhost:9200'],
    basic_auth=('elastic', 'QANe83Fe+mfgKSOd94Zv'),
    ssl_context=ssl_context
)

# Vérification de la connexion Elasticsearch
if es.ping():
    print("Connexion réussie à Elasticsearch.")
else:
    print("Impossible de se connecter à Elasticsearch. Vérifiez l'URL et les paramètres SSL.")

# Fonction pour charger les documents
def load_documents(path):
    loader = DirectoryLoader(path, glob="*.pdf")
    documents = loader.load()
    return documents

documents = load_documents(PATH)

# Définition du text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=200,
    separators=["\n", "\n\n", "(?<=\.!-/)","", " "]
)

# Division des documents en données
data = text_splitter.split_documents(documents)

# Initialisation de l'embedding
embedding = OpenAIEmbeddings()

# Configuration de l'URL Elasticsearch
user_name = "elastic"
password = 'QANe83Fe+mfgKSOd94Zv'
elasticsearch_url = f"https://{user_name}:{password}@localhost:9200"

# Instanciation de ElasticVectorSearch
elastic_vector_search = ElasticsearchStore.from_documents(
    elasticsearch_url=elasticsearch_url,
    documents=data,
    index_name="test_index",
    es_connection=es,
    embedding=embedding,
)

"""
db = ElasticsearchStore.from_documents(
    elasticsearch_url=elasticsearch_url,
    index_name='vectors',
    documents=data,
    embedding=embedding
)"""
