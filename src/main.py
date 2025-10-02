# app_gradio.py
import logging
import gradio as gr
import os

from storage import create_storage
from pypdf import PdfReader

from embeddings import get_embedding
from llm import rag_chat

# NEW: import the Gmail tab helper
from gmail_ingest import add_gmail_tab

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
INDEX_NAME = "personal_documents"

# Initialize Elasticsearch
ES_HOST = os.getenv("ES_HOST", "elasticsearch")
ES_PORT = os.getenv("ES_PORT", "9200")
ES_USERNAME = os.getenv('ES_USERNAME', 'elastic')
ES_PASSWORD = os.getenv('ES_PASSWORD', 'changeme')
storage = create_storage(
    "elasticsearch",
    host=f"http://{ES_HOST}:{ES_PORT}",
    username=ES_USERNAME,
    password=ES_PASSWORD,
    index_name=INDEX_NAME,
    embedding_dim=768
)

def process_file(files):
    """Process uploaded files"""
    if not files:
        return "Please upload files first."

    results = []
    for file in files:
        try:
            content = ""
            filename = os.path.basename(file.name)
            if filename.lower().endswith('.pdf'):
                # Use PyPDF to read PDF files
                reader = PdfReader(file)
                for page in reader.pages:
                    content += page.extract_text() + "\n"
            else:
                with open(file.name, "r", encoding="utf-8") as f:
                    content = f.read()

            result = storage.store_document(
                content=content,
                filename=filename,
                get_embedding_fn=get_embedding,
            )
            results.append(f"✅ {filename}: {result}")
            logger.info(f"Processed {filename}: {result}")
            
        except Exception as e:
            error_msg = f"❌ Error processing {filename}: {str(e)}"
            results.append(error_msg)
            logger.error(error_msg)

    return "\n".join(results)

def process_single_file(file):
    """Wrapper for backward compatibility with single file processing"""
    if file:
        return process_file([file])
    return "Please upload a file first."

# NEW: adapter to store one email into your index
def store_email_document(email_doc: dict) -> str:
    """
    Adapts a Gmail email_doc into your storage format.
    Uses storage.store_document(get_embedding_fn=get_embedding).
    Returns a short status string for the UI log.
    """
    subject = email_doc.get("subject") or "(no subject)"
    from_field = email_doc.get("from") or ""
    dt = email_doc.get("date") or ""
    header_block = f"Subject: {subject}\nFrom: {from_field}\nDate: {dt}\n\n"
    body = email_doc.get("text") or ""
    content = header_block + body

    # Stable key to avoid duplication if re-ingested
    filename = f"gmail:{email_doc['id']}"

    res = storage.store_document(
        content=content,
        filename=filename,
        get_embedding_fn=get_embedding,
    )
    return f"OK {subject[:80]} — {res}"

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Personal RAG Assistant") as app:
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
        file_output = gr.Textbox(label="Status", lines=10)
        with gr.Row():
            file_input = gr.File(
                label="Upload Documents", 
                file_types=[".txt", ".pdf"],
                file_count="multiple"  # Enable multiple file selection
            )
            upload_btn = gr.Button("Upload and Index")
    
    with gr.Tab("Document Search"):
        search_query = gr.Textbox(label="Search documents")
        search_results = gr.JSON(label="Search Results")
        search_btn = gr.Button("Search")
    
    # Event handlers
    def respond(message, chat_history, chunks):
        chat_history.append([message, ""])
        for response in rag_chat(message, chat_history, int(chunks), storage.search_similar, get_embedding):
            chat_history[-1][1] = response
            yield chat_history, ""
    
    submit.click(respond, [msg, chatbot, num_chunks], [chatbot, msg])
    msg.submit(respond, [msg, chatbot, num_chunks], [chatbot, msg])
    clear.click(lambda: None, None, chatbot, queue=False)
    
    upload_btn.click(process_file, file_input, file_output)
    
    def search_docs(query, search_type="Combined"):
        if search_type == "Filename (text)":
            return storage._search_filename_text(query, k=5)
        elif search_type == "Content (semantic)":
            return storage._search_by_content(query, get_embedding, k=5)
        elif search_type == "Filename (semantic)":
            return storage._search_by_filename(query, get_embedding, k=5)
        elif search_type == "Hybrid":
            return storage.hybrid_search(query, get_embedding, k=5)
        else:  # Combined (default)
            return storage.search_similar(query, get_embedding, k=5, search_type="combined")

    search_btn.click(search_docs, search_query, search_results)

    # --- New Gmail tab (mounted without passing the Blocks object) ---
    add_gmail_tab(store_email_fn=store_email_document, default_query="in:inbox newer_than:30d")

if __name__ == "__main__":
    if storage.initialize():
        logger.info("Storage initialized successfully.")
        app.launch(server_name="0.0.0.0", server_port=7860, share=False)
    else:
        logger.error("Failed to initialize storage. Exiting.")
