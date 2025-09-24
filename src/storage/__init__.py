import os
import logging

from .storage import VectorStorage
from .elastic import ElasticsearchStorage

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Factory function for creating storage instances
def create_storage(storage_type: str, **kwargs) -> VectorStorage:
    """
    Factory function to create storage instances
    
    Args:
        storage_type: Type of storage ('elasticsearch', 'chroma', 'pinecone')
        **kwargs: Configuration parameters for the storage
        
    Returns: VectorStorage instance
    """
    if storage_type == "elasticsearch":
        from elasticsearch import Elasticsearch
        es_client = Elasticsearch(
            kwargs.get("host", "http://localhost:9200"),
            basic_auth=(
                kwargs.get("username", "elastic"),
                kwargs.get("password", "changeme")
            ),
            verify_certs=kwargs.get("verify_certs", False)
        )
        return ElasticsearchStorage(
            index_name=kwargs.get("index_name", "documents"),
            es_client=es_client,
            embedding_dim=kwargs.get("embedding_dim", 768)
        )
    
    # Add other storage types here
    # elif storage_type == "chroma":
    #     return ChromaStorage(**kwargs)
    # elif storage_type == "pinecone":
    #     return PineconeStorage(**kwargs)
    
    else:
        raise ValueError(f"Unsupported storage type: {storage_type}")
