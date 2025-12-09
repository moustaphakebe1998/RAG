import os
import ssl
import urllib3
#import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)



#ca_cert_path = '/Users/moustaphakeb/airflow-project/key/http_ca.crt'

#ssl_context = ssl.create_default_context(cafile=ca_cert_path)
#ssl_context.check_hostname = False
#ssl_context.verify_mode = ssl.CERT_NONE




def test_es_connection():
    ES_URL = "https://elastic:Z25ft0VLU2fpiqOXRSEc@localhost:9200"
    
    es = Elasticsearch(
        ES_URL,
        verify_certs=False,   # ignore le certificat SSL
        ssl_show_warn=False   # supprime les warnings SSL
    )

    if es.ping():
        print("Connexion réussie à Elasticsearch !")
    else:
        raise Exception("Impossible de se connecter à Elasticsearch.")
    

if __name__ == "__main__":
    test_es_connection()