# app_gradio.py
import logging
import gradio as gr
import os

from embeddings import get_embedding
from storage import search_similar_chunks, index_document, wait_for_elasticsearch_ready, create_index
from llm import rag_chat

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
INDEX_NAME = "personal_documents"
def process_file(file):
    """Process uploaded file"""
    if file:
        with open(file.name, "r", encoding="utf-8") as f:
            content = f.read()
        result = index_document(content, os.path.basename(file.name), INDEX_NAME, get_embedding)
        return result
    return "Please upload a file first."

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Personal RAG Assistant") as demo:
    gr.Markdown("# 📚 Personal RAG Assistant")
    gr.Markdown("Upload your documents and chat with them using AI!")
    
    with gr.Tab("Chat"):
        chatbot = gr.Chatbot(height=500)
        with gr.Row():
            msg = gr.Textbox(
                label="Type your question",
                placeholder="Ask anything about your documents...",
                scale=4
            )
            submit = gr.Button("Send", variant="primary")
        
        with gr.Row():
            clear = gr.Button("Clear Chat")
            num_chunks = gr.Slider(1, 5, value=3, label="Context chunks")
    
    with gr.Tab("Upload Documents"):
        file_output = gr.Textbox(label="Status")
        with gr.Row():
            file_input = gr.File(label="Upload Document", file_types=[".txt", ".pdf", ".md"])
            upload_btn = gr.Button("Upload and Index")
    
    with gr.Tab("Document Search"):
        search_query = gr.Textbox(label="Search documents")
        search_results = gr.JSON(label="Search Results")
        search_btn = gr.Button("Search")
    
    # Event handlers
    def respond(message, chat_history, chunks):
        chat_history.append([message, ""])
        for response in rag_chat(message, chat_history, int(chunks), search_similar_chunks, get_embedding, INDEX_NAME):
            chat_history[-1][1] = response
            yield chat_history, ""
    
    submit.click(respond, [msg, chatbot, num_chunks], [chatbot, msg])
    msg.submit(respond, [msg, chatbot, num_chunks], [chatbot, msg])
    clear.click(lambda: None, None, chatbot, queue=False)
    
    upload_btn.click(process_file, file_input, file_output)
    
    def search_docs(query):
        results = search_similar_chunks(query, get_embedding, INDEX_NAME, k=5)
        return results
    
    search_btn.click(search_docs, search_query, search_results)

if __name__ == "__main__":
    if not wait_for_elasticsearch_ready(timeout=180):
        raise SystemExit("Elasticsearch service is not available. Exiting.")

    create_index(index_name=INDEX_NAME)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)