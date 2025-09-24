import os
import time
import logging

from elasticsearch import Elasticsearch
from typing import List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Elasticsearch
ES_HOST = os.getenv("ES_HOST", "elasticsearch")
ES_PORT = os.getenv("ES_PORT", "9200")
ES_USERNAME = os.getenv('ELASTICSEARCH_USERNAME', 'elastic')
ES_PASSWORD = os.getenv('ELASTICSEARCH_PASSWORD', 'changeme123!')
es = Elasticsearch(
    f"http://{ES_HOST}:{ES_PORT}",
    basic_auth=(ES_USERNAME, ES_PASSWORD),
    verify_certs=False,
)

def wait_for_elasticsearch_ready(es_client = es, timeout=120):
    """Wait for Elasticsearch to be fully ready for operations"""
    logger.info("Waiting for Elasticsearch to be fully ready...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            # Check if cluster is ready
            health = es_client.cluster.health()
            if health['status'] in ['yellow', 'green']:
                logger.info(f"✅ Elasticsearch is ready! Status: {health['status']}")
                return True
        except ConnectionError:
            logger.debug("Elasticsearch not ready yet...")
        except Exception as e:
            logger.debug(f"Elasticsearch not fully ready: {e}")
        
        time.sleep(5)
    
    logger.error("❌ Elasticsearch failed to become ready within timeout")
    return False

def create_index(index_name: str) -> bool:
    """Create Elasticsearch index with vector mapping"""
    if not es.indices.exists(index=index_name):
        mapping = {
            "mappings": {
                "properties": {
                    "content": {"type": "text"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 768,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "filename": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    "timestamp": {"type": "date"}
                }
            }
        }
        es.indices.create(index=index_name, body=mapping)
    return True

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks

def index_document(file_content: str, filename: str, index_name, get_embedding) -> str:
    """Index document in Elasticsearch"""
    chunks = chunk_text(file_content)
    
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        if embedding:
            doc = {
                "content": chunk,
                "embedding": embedding,
                "filename": filename,
                "chunk_index": i,
                "timestamp": time.time() * 1000
            }
            es.index(index=index_name, document=doc)
    
    return f"Indexed {len(chunks)} chunks from {filename}"

def search_similar_chunks(query: str, get_embedding, index_name, k: int = 3) -> List[Dict]:
    """Search for similar chunks using vector search"""
    query_embedding = get_embedding(query)
    if not query_embedding:
        return []
    
    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                "params": {"query_vector": query_embedding}
            }
        }
    }
    
    try:
        response = es.search(
            index=index_name,
            body={"size": k, "query": script_query, "_source": ["content", "filename"]}
        )
        
        return [{
            "content": hit["_source"]["content"],
            "filename": hit["_source"]["filename"],
            "score": hit["_score"]
        } for hit in response["hits"]["hits"]]
    except Exception as e:
        print(f"Search error: {e}")
        return []

