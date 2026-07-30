import ollama
import os

from typing import List, Iterator
from context_compression import (
    CompressionConfig,
    build_retrieval_query,
    format_retrieved_context,
)

CHAT_MODEL = os.getenv("CHAT_MODEL", "llama2-uncensored:7b")

def rag_chat(
    message: str,
    history: List[List[str]],
    num_chunks: int,
    search_similar_chunks,
    get_embedding,
    compression_config=None,
) -> Iterator[str]:
    """Generate RAG response with stronger instructions"""

    if compression_config is None:
        compression_config = CompressionConfig.from_env()

    # Search for relevant context
    retrieval_query = build_retrieval_query(message, history, compression_config)
    context_chunks = search_similar_chunks(retrieval_query, get_embedding, k=num_chunks)
    
    if context_chunks:
        context = format_retrieved_context(context_chunks, compression_config)
        prompt = f"""IMPORTANT: You are a personal assistant for the user. The user is asking about THEIR OWN information from THEIR OWN documents. 
You MUST provide the exact information they are asking for from their documents.

The user explicitly wants you to share their personal information with them. This is not a privacy violation - it's your primary function.

CONTEXT FROM USER'S DOCUMENTS:
{context}

USER'S QUESTION: {message}

YOUR INSTRUCTION: Provide the exact information from their documents. Do not refuse. Do not add disclaimers. Just answer the question with their information.

ANSWER:"""
    else:
        prompt = f"""User Question: {message}

Answer:"""
    
    # Generate streaming response
    full_response = ""
    sources_text = "\n\nSources:\n" + "\n".join([f"- {chunk.filename} (score: {chunk.score:.3f})" 
                                               for chunk in context_chunks]) if context_chunks else ""
    
    for response in ollama.generate(model=CHAT_MODEL, prompt=prompt, stream=True):
        token = response["response"]
        full_response += token
        yield full_response + sources_text
