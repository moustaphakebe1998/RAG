# RAG
Dans la partie Elasticsearch on peut trouver toutes les resources nécessaires sur **https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html** 

pour creer ton image docker avec elasticsearch"

docker pull docker.elastic.co/elasticsearch/elasticsearch:8.12.2

docker network create elastic

docker run --name es01 --net elastic -p 9200:9200 -it -m 1GB docker.elastic.co/elasticsearch/elasticsearch:8.12.2

docker exec -it es01 /usr/share/elasticsearch/bin/elasticsearch-reset-password -u elastic

export ELASTIC_PASSWORD="your_password"

docker cp es01:/usr/share/elasticsearch/config/certs/http_ca.crt .

curl --cacert http_ca.crt -u elastic:$ELASTIC_PASSWORD https://localhost:9200

# Remplacez 'votre_index' par le nom de votre index Elasticsearch

curl -u elastic:votre_mot_de_passe -k -X DELETE "https://localhost:9200/votre_index"

**exemple dans mon cas:**

export ELASTIC_PASSWORD='QANe83Fe+mfgKSOd94Zv'
curl --cacert /home/kebe/langchain/elasticshearch/http_ca.crt -u elastic:$ELASTIC_PASSWORD https://localhost:9200/file_chatbot


curl -X GET "https://localhost:9200/file_chatbot/_search" --cacert /home/kebe/langchain/elasticshearch/http_ca.crt -u elastic:$ELASTIC_PASSWORD -H 'Content-Type: application/json' -d'

1) premier index
   
{
  "query": {
    "match": {
      "page_content": "Une subvention de fonctionnement"
    }
  }
}
'
2) deuxieme indexes
 curl -X GET "https://localhost:9200/test_index/_search" --cacert /home/kebe/langchain/elasticshearch/http_ca.crt -u elastic:$ELASTIC_PASSWORD -H 'Content-Type: application/json' -d'
{
  "size": 10,
  "_source": ["text"],
  "query": {
    "match": {
      "text": "Une subvention de fonctionnement"
    }
  }
}
