import os
import ssl
import urllib3
from elasticsearch import Elasticsearch
from langchain_elasticsearch import ElasticsearchRetriever
from langchain import llms
from langchain.memory import ElasticsearchChatMessageHistory

from langchain_community.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
# Conversational chain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# Streaming
from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)
from langchain.retrievers.self_query.base import SelfQueryRetriever

# URL de base de votre instance Elasticsearch
es_base_url = "http://localhost:9200"
# Nom de l'index où sont stockés vos documents et vecteurs
index_name = "test_index"
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
user_name="elastic"
password='QANe83Fe+mfgKSOd94Zv'
elasticsearch_url = f"https://{user_name}:{password}@localhost:9200"
# Création du retriever Elasticsearch à partir des données récupérées depuis votre instance Elasticsearch
es_client = es
def body_func(text):
    return {
        "query": {
            "match": {
                "text": text  # Champ contenant le texte des documents dans votre index Elasticsearch
            }
        }
    }

# Création du retriever Elasticsearch
retriever = ElasticsearchRetriever(
    index_name="test_index",  # Nom de l'index Elasticsearch
    elasticsearch_url=elasticsearch_url,  # URL de votre instance Elasticsearch
    es_client=es,  # Client Elasticsearch
    body_func=body_func,  # Fonction de génération de la requête Elasticsearch
    embedding=None  # L'embedding n'est pas nécessaire car vous avez déjà des vecteurs dans vos documents
)
# Création du modèle de langage naturel (LLM)
llm = ChatOpenAI(model="gpt-4")  # Utilisez le modèle de langage naturel de votre choix

# Création du ConversationalRetrievalChain
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=True  # Indique si les documents source doivent être renvoyés avec les réponses
)
chain("les subventions de la dac ?")




