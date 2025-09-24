import time
from typing import Dict, Any, List, Optional
from .storage import VectorStorage, DocumentChunk, SearchResult

class ElasticsearchStorage(VectorStorage):
    """Elasticsearch implementation with enhanced search capabilities"""
    
    def __init__(self, 
                 index_name: str, 
                 es_client: Any,
                 embedding_dim: int = 768,
                 **kwargs):
        super().__init__(index_name, embedding_dim, **kwargs)
        self.es_client = es_client
    
    def initialize(self) -> bool:
        """Create Elasticsearch index with enhanced mapping for filename search"""
        try:
            if not self.es_client.indices.exists(index=self.index_name):
                mapping = {
                    "mappings": {
                        "properties": {
                            "content": {"type": "text"},
                            "filename": {
                                "type": "text",  # Changed to text for full-text search
                                "fields": {
                                    "keyword": {"type": "keyword"}  # Keep keyword for exact matches
                                }
                            },
                            "filename_embedding": {  # NEW: Separate embedding for filename
                                "type": "dense_vector",
                                "dims": self.embedding_dim,
                                "index": True,
                                "similarity": "cosine"
                            },
                            "content_embedding": {  # RENAMED: Clearer distinction
                                "type": "dense_vector",
                                "dims": self.embedding_dim,
                                "index": True,
                                "similarity": "cosine"
                            },
                            "combined_embedding": {  # NEW: Content + filename embedding
                                "type": "dense_vector", 
                                "dims": self.embedding_dim,
                                "index": True,
                                "similarity": "cosine"
                            },
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
                      chunk_size: int = 500, overlap: int = 50, 
                      include_filename_in_search: bool = True) -> str:
        """Store document with optional filename inclusion in search"""
        chunks = self.chunk_text(content, chunk_size, overlap)
        
        for chunk in chunks:
            chunk.filename = filename
            
            # Generate different embedding strategies
            chunk.content_embedding = get_embedding_fn(chunk.content)
            chunk.filename_embedding = get_embedding_fn(filename)
            
            # Combined embedding for hybrid search
            if include_filename_in_search:
                combined_text = f"Filename: {filename}\nContent: {chunk.content}"
                chunk.combined_embedding = get_embedding_fn(combined_text)
            else:
                chunk.combined_embedding = chunk.content_embedding
        
        stored_count = self.store_chunks(chunks, get_embedding_fn)
        return f"Indexed {stored_count} chunks from {filename}"
    
    def store_chunks(self, chunks: List[DocumentChunk], get_embedding_fn: callable) -> int:
        """Store chunks with multiple embedding strategies"""
        stored_count = 0
        
        for chunk in chunks:
            try:
                doc = {
                    "content": chunk.content,
                    "filename": chunk.filename,
                    "content_embedding": chunk.content_embedding,
                    "filename_embedding": getattr(chunk, 'filename_embedding', []),
                    "combined_embedding": getattr(chunk, 'combined_embedding', chunk.content_embedding),
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
                      k: int = 3, filters: Optional[Dict[str, Any]] = None,
                      search_type: str = "combined") -> List[SearchResult]:
        """Enhanced search with multiple strategies"""
        
        if search_type == "filename_only":
            return self._search_by_filename(query, k, filters)
        elif search_type == "content_only":
            return self._search_by_content(query, get_embedding_fn, k, filters)
        elif search_type == "filename_text":
            return self._search_filename_text(query, k, filters)
        else:  # combined or default
            return self._search_combined(query, get_embedding_fn, k, filters)
    
    def _search_by_content(self, query: str, get_embedding_fn: callable, 
                          k: int = 3, filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search by content embedding only"""
        query_embedding = get_embedding_fn(query)
        return self._vector_search(query_embedding, "content_embedding", k, filters)
    
    def _search_by_filename(self, query: str, get_embedding_fn: callable, 
                           k: int = 3, filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search by filename embedding only"""
        query_embedding = get_embedding_fn(query)
        return self._vector_search(query_embedding, "filename_embedding", k, filters)
    
    def _search_combined(self, query: str, get_embedding_fn: callable, 
                        k: int = 3, filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search using combined content + filename embedding"""
        query_embedding = get_embedding_fn(query)
        return self._vector_search(query_embedding, "combined_embedding", k, filters)
    
    def _search_filename_text(self, query: str, k: int = 3, 
                            filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Traditional text search on filename"""
        try:
            # Build base query
            base_query = {"match": {"filename": query}}
            
            # Add filters if provided
            if filters:
                base_query = {
                    "bool": {
                        "must": [base_query],
                        "filter": [{"term": {k: v}} for k, v in filters.items()]
                    }
                }
            
            response = self.es_client.search(
                index=self.index_name,
                body={
                    "size": k, 
                    "query": base_query,
                    "sort": [{"_score": {"order": "desc"}}],
                    "_source": True
                }
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
            print(f"Filename text search error: {e}")
            return []
    
    def _vector_search(self, query_embedding: List[float], field: str, 
                      k: int = 3, filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Generic vector search implementation"""
        if not query_embedding:
            return []
        
        # Build filter query
        filter_query = {"match_all": {}}
        if filters:
            filter_query = {"bool": {"must": []}}
            for field_name, value in filters.items():
                filter_query["bool"]["must"].append({"term": {field_name: value}})
        
        script_query = {
            "script_score": {
                "query": filter_query,
                "script": {
                    "source": f"cosineSimilarity(params.query_vector, '{field}') + 1.0",
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
            print(f"Vector search error ({field}): {e}")
            return []
    
    def hybrid_search(self, query: str, get_embedding_fn: callable, 
                     k: int = 3, content_weight: float = 0.7, 
                     filename_weight: float = 0.3) -> List[SearchResult]:
        """Combine vector and text search for best results"""
        # Get content-based results
        content_results = self._search_by_content(query, get_embedding_fn, k*2)
        
        # Get filename-based results
        filename_results = self._search_filename_text(query, k*2)
        
        # Combine and re-rank
        all_results = {}
        for result in content_results:
            key = result.chunk_id
            if key not in all_results:
                all_results[key] = result
                all_results[key].score = result.score * content_weight
            else:
                all_results[key].score += result.score * content_weight
        
        for result in filename_results:
            key = result.chunk_id
            if key not in all_results:
                all_results[key] = result
                all_results[key].score = result.score * filename_weight
            else:
                all_results[key].score += result.score * filename_weight
        
        # Return top k results
        sorted_results = sorted(all_results.values(), key=lambda x: x.score, reverse=True)
        return sorted_results[:k]

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
