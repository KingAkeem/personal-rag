import ollama
import os

from typing import List, Generator
from typing import Iterator

CHAT_MODEL = os.getenv("CHAT_MODEL", "llama2:7b")
def rag_chat(
    message: str,
    history: List[List[str]],
    num_chunks: int,
    search_similar_chunks,
    get_embedding,
    INDEX_NAME
) -> Iterator[str]:
    """Generate RAG response with streaming"""
    # Search for relevant context
    context_chunks = search_similar_chunks(message, get_embedding, INDEX_NAME, k=num_chunks)
    
    if context_chunks:
        context = "\n\n".join([f"From {chunk['filename']}:\n{chunk['content']}" 
                              for chunk in context_chunks])
        prompt = f"""Based on the following context, answer the user's question. 
If the context doesn't contain relevant information, say so.

Context:
{context}

User Question: {message}

Answer:"""
    else:
        prompt = f"""User Question: {message}

Answer:"""
    
    # Generate streaming response
    full_response = ""
    sources_text = "\n\nSources:\n" + "\n".join([f"- {chunk['filename']} (score: {chunk['score']:.3f})" 
                                               for chunk in context_chunks]) if context_chunks else ""
    
    for response in ollama.generate(model=CHAT_MODEL, prompt=prompt, stream=True):
        token = response["response"]
        full_response += token
        yield full_response + sources_text
