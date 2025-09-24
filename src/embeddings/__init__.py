import ollama
import os

from typing import List

EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', "nomic-embed-text")
def get_embedding(text: str) -> List[float]:
    """Get embedding vector from Ollama"""
    try:
        response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
        return response["embedding"]
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return []


__all__ = ["get_embedding"]