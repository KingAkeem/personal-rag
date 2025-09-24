import time

from typing import Dict, Any, List, Optional
from .storage import VectorStorage, DocumentChunk, SearchResult

# Concrete implementation example for Elasticsearch
class ElasticsearchStorage(VectorStorage):
    """Elasticsearch implementation of VectorStorage"""
    
    def __init__(self, 
                 index_name: str, 
                 es_client: Any,
                 embedding_dim: int = 768,
                 **kwargs):
        super().__init__(index_name, embedding_dim, **kwargs)
        self.es_client = es_client
    
    def initialize(self) -> bool:
        """Create Elasticsearch index with vector mapping"""
        try:
            if not self.es_client.indices.exists(index=self.index_name):
                mapping = {
                    "mappings": {
                        "properties": {
                            "content": {"type": "text"},
                            "embedding": {
                                "type": "dense_vector",
                                "dims": self.embedding_dim,
                                "index": True,
                                "similarity": "cosine"
                            },
                            "filename": {"type": "keyword"},
                            "chunk_index": {"type": "integer"},
                            "chunk_id": {"type": "keyword"},
                            "metadata": {"type": "object"},
                            "timestamp": {"type": "date"}
                        }
                    }
                }
                self.es_client.indices.create(index=self.index_name, body=mapping)
            
            self._initialized = True
            return True
        except Exception as e:
            print(f"Error initializing Elasticsearch index: {e}")
            return False
    
    def store_document(self, content: str, filename: str, get_embedding_fn: callable, 
                      chunk_size: int = 500, overlap: int = 50) -> str:
        """Store document in Elasticsearch"""
        chunks = self.chunk_text(content, chunk_size, overlap)
        
        for chunk in chunks:
            chunk.filename = filename
            chunk.embedding = get_embedding_fn(chunk.content)
        
        stored_count = self.store_chunks(chunks, get_embedding_fn)
        return f"Indexed {stored_count} chunks from {filename}"
    
    def store_chunks(self, chunks: List[DocumentChunk], get_embedding_fn: callable) -> int:
        """Store chunks in Elasticsearch"""
        stored_count = 0
        
        for chunk in chunks:
            try:
                # Generate embedding if not already provided
                if chunk.embedding is None:
                    chunk.embedding = get_embedding_fn(chunk.content)
                
                doc = {
                    "content": chunk.content,
                    "embedding": chunk.embedding,
                    "filename": chunk.filename,
                    "chunk_index": chunk.chunk_index,
                    "chunk_id": chunk.chunk_id,
                    "metadata": chunk.metadata,
                    "timestamp": time.time() * 1000
                }
                
                self.es_client.index(index=self.index_name, document=doc)
                stored_count += 1
            except Exception as e:
                print(f"Error storing chunk {chunk.chunk_id}: {e}")
        
        return stored_count
    
    def search_similar(self, query: str, get_embedding_fn: callable, 
                      k: int = 3, filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search for similar chunks using vector similarity"""
        query_embedding = get_embedding_fn(query)
        if not query_embedding:
            return []
        
        # Build filter query if filters provided
        filter_query = {"match_all": {}}
        if filters:
            filter_query = {"bool": {"must": []}}
            for field, value in filters.items():
                filter_query["bool"]["must"].append({"term": {field: value}})
        
        script_query = {
            "script_score": {
                "query": filter_query,
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": query_embedding}
                }
            }
        }
        
        try:
            response = self.es_client.search(
                index=self.index_name,
                body={"size": k, "query": script_query, "_source": True}
            )
            
            return [SearchResult(
                content=hit["_source"]["content"],
                filename=hit["_source"]["filename"],
                score=hit["_score"],
                chunk_index=hit["_source"].get("chunk_index", 0),
                metadata=hit["_source"].get("metadata", {}),
                chunk_id=hit["_source"].get("chunk_id")
            ) for hit in response["hits"]["hits"]]
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def delete_document(self, filename: str) -> bool:
        """Delete all chunks of a specific document"""
        try:
            response = self.es_client.delete_by_query(
                index=self.index_name,
                body={"query": {"term": {"filename": filename}}}
            )
            return response["deleted"] > 0
        except Exception as e:
            print(f"Error deleting document {filename}: {e}")
            return False
    
    def get_document_chunks(self, filename: str) -> List[DocumentChunk]:
        """Retrieve all chunks of a specific document"""
        try:
            response = self.es_client.search(
                index=self.index_name,
                body={
                    "query": {"term": {"filename": filename}},
                    "sort": [{"chunk_index": {"order": "asc"}}],
                    "size": 1000
                }
            )
            
            return [DocumentChunk(
                content=hit["_source"]["content"],
                filename=hit["_source"]["filename"],
                chunk_index=hit["_source"]["chunk_index"],
                embedding=hit["_source"].get("embedding"),
                metadata=hit["_source"].get("metadata", {}),
                chunk_id=hit["_source"].get("chunk_id")
            ) for hit in response["hits"]["hits"]]
        except Exception as e:
            print(f"Error retrieving document chunks: {e}")
            return []
    
    def health_check(self) -> bool:
        """Check if Elasticsearch is healthy"""
        try:
            return self.es_client.ping()
        except:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Elasticsearch index statistics"""
        try:
            stats = self.es_client.indices.stats(index=self.index_name)
            count = self.es_client.count(index=self.index_name)
            
            return {
                "document_count": count["count"],
                "index_size_bytes": stats["indices"][self.index_name]["total"]["store"]["size_in_bytes"],
                "health": "healthy" if self.health_check() else "unhealthy"
            }
        except Exception as e:
            return {"error": str(e)}
