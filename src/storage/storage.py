import hashlib

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class DocumentChunk:
    """Represents a chunk of a document with metadata"""
    content: str
    filename: str
    chunk_index: int
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = None
    chunk_id: str = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.chunk_id is None:
            # Generate a unique ID for the chunk
            content_hash = hashlib.md5(self.content.encode()).hexdigest()
            self.chunk_id = f"{self.filename}_{self.chunk_index}_{content_hash[:8]}"

@dataclass
class SearchResult:
    """Represents a search result with similarity score"""
    content: str
    filename: str
    score: float
    chunk_index: int
    metadata: Dict[str, Any] = None
    chunk_id: str = None

class VectorStorage(ABC):
    """
    Abstract base class for vector storage implementations.
    Supports different vector databases (Elasticsearch, Chroma, Pinecone, etc.)
    """
    
    def __init__(self, index_name: str, embedding_dim: int = 768, **kwargs):
        self.index_name = index_name
        self.embedding_dim = embedding_dim
        self._initialized = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the storage (create index/collection if needed)
        Returns: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def store_document(self, 
                      content: str, 
                      filename: str, 
                      get_embedding_fn: callable,
                      chunk_size: int = 500,
                      overlap: int = 50) -> str:
        """
        Store a document by chunking it and generating embeddings
        
        Args:
            content: The text content of the document
            filename: Name of the file
            get_embedding_fn: Function to generate embeddings
            chunk_size: Size of each text chunk
            overlap: Overlap between chunks
            
        Returns: Status message
        """
        pass
    
    @abstractmethod
    def store_chunks(self, 
                    chunks: List[DocumentChunk],
                    get_embedding_fn: callable) -> int:
        """
        Store multiple document chunks with embeddings
        
        Args:
            chunks: List of DocumentChunk objects
            get_embedding_fn: Function to generate embeddings
            
        Returns: Number of chunks successfully stored
        """
        pass
    
    @abstractmethod
    def search_similar(self, 
                      query: str, 
                      get_embedding_fn: callable,
                      k: int = 3,
                      filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Search for similar chunks using vector similarity
        
        Args:
            query: Search query text
            get_embedding_fn: Function to generate query embedding
            k: Number of results to return
            filters: Optional filters (e.g., {"filename": "specific_file.txt"})
            
        Returns: List of SearchResult objects
        """
        pass
    
    @abstractmethod
    def delete_document(self, filename: str) -> bool:
        """
        Delete all chunks of a specific document
        
        Args:
            filename: Name of the file to delete
            
        Returns: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_document_chunks(self, filename: str) -> List[DocumentChunk]:
        """
        Retrieve all chunks of a specific document
        
        Args:
            filename: Name of the document
            
        Returns: List of DocumentChunk objects
        """
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if the storage backend is healthy
        
        Returns: True if healthy, False otherwise
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the storage
        
        Returns: Dictionary with statistics
        """
        pass
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[DocumentChunk]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text to chunk
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            
        Returns: List of DocumentChunk objects
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_content = " ".join(words[i:i + chunk_size])
            chunk = DocumentChunk(
                content=chunk_content,
                filename="",  # Will be set by caller
                chunk_index=len(chunks)
            )
            chunks.append(chunk)
            
            if i + chunk_size >= len(words):
                break
                
        return chunks
    
    def wait_for_ready(self, timeout: int = 120) -> bool:
        """
        Wait for the storage backend to be ready
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns: True if ready within timeout, False otherwise
        """
        # Default implementation - can be overridden by subclasses
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.health_check():
                return True
            time.sleep(5)
        
        return False