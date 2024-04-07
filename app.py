import os
import json
import streamlit as st
import pandas as pd
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain import llms
import chromadb.config
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


Data_PATH="DocsRAG"
embedding= OpenAIEmbeddings()
def load_document(PATH):
    #loader = DirectoryLoader(Data_PATH, glob="*.pdf")
    loader = DirectoryLoader(PATH, glob="*.pdf")
    documents = loader.load()
    return documents

documents=load_document(Data_PATH)

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=200,
    separators=["\n","\n\n","(?<=\.!-/)",""," "]
)
chunks=text_splitter.split_documents(documents)

CHROMA_PATH='home/kebe/database'
#Creation de la data base chroma
dbase=Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory=CHROMA_PATH
)
llm = llms.OpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0)
chat_llm = ChatOpenAI(streaming=True,model="gpt-4",  callbacks=[StreamingStdOutCallbackHandler()], temperature=0,max_tokens=2500)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
#template
template = """
Utilisez les éléments de contexte suivants pour répondre à la question à la fin.
Si vous ne connaissez pas la réponse, veuillez simplement indiquer que vous ne savez pas. Vous êtes un agent des services municipaux de Paris, veuillez vous en tenir aux informations officielles.
CONTEXTE : {context}
-------
EXTRAIT DES BULLETINS OFFICIELS DES DÉLIBÉRATIONS DE LA VILLE DE PARIS :
{chat_history}
Humain : {question}
Assistant :

"""
#Prompts
prompts=PromptTemplate(input_variables=["context", "question", "chat_history"], template=template)
# Configuration du modèle de chaîne de récupération conversationnelle
model_simple= ConversationalRetrievalChain.from_llm(chat_llm,
                                           dbase.as_retriever(),
                                           return_source_documents=True, memory=memory,
                                              combine_docs_chain_kwargs={"prompt": prompts})


def chatbot_response(input_text):
    return model_simple(input_text)

# Fonction pour obtenir une réponse du chatbot (à définir selon votre modèle)
# Titre de l'application web
st.title("Posez votre question au chatbot")
chat_history = []
with st.container():
    # Champ de saisie pour la question de l'utilisateur
    query = st.text_input("Que souhaitez-vous demander ?", key="query")

    # Bouton pour soumettre la question et obtenir une réponse
    if st.button("Soumettre", key="submit"):
        if not query.strip():
            # Afficher une erreur si la question est vide
            st.error("Veuillez fournir une question.")
        else:
            try:
                # Obtenir une réponse à partir du chatbot
                response = chatbot_response(query)
                response_text = response["answer"]
                # Afficher la réponse directement sous la question
                st.markdown(response_text)
                # Ajouter la question à l'historique de chat
                chat_history.append(query)
                # Ajouter la réponse à l'historique de chat
                chat_history.append(response_text)
                # Source des documents auxquels le Chat a obtenu ses réponses
                source_docs = response["source_documents"]
                # Affichage des réponses
                for doc in source_docs:
                    st.markdown(doc.page_content)
            except Exception as e:
                # Gérer les exceptions et afficher les erreurs
                st.error(f"Une erreur s'est produite: {e}")

# Afficher l'historique de chat dans la partie gauche après interaction
if chat_history:
    with st.sidebar:
        st.subheader("Historique de Chat")
        for message in chat_history:
            st.markdown(message)

#Afficher l'historique des chats juste la premiére élément 
#st.subheader("Historique de Chat")
#response_history = response["chat_history"][0]
#st.markdown(response_history.content)
#Chat history
#chat_history.append(query)
# Ajouter la réponse à l'historique de chat
#chat_history.append(response_text)
