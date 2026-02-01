import gradio as gr
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pypdf import PdfReader
import os
import ollama

# === CONFIG ===
MODEL_NAME = 'all-MiniLM-L6-v2'  # Fast, good embedding model
DOCUMENTS_FOLDER = "documents"   # Your files go here
CHUNK_SIZE = 500                 # Characters per chunk
CHUNK_OVERLAP = 100              # Overlap for context
TOP_K = 3                        # Retrieve top 3 chunks

# Load embedding model (runs locally)
embedder = SentenceTransformer(MODEL_NAME)

# Global variables for vector store
index = None
chunks = []
metadata = []

def load_documents():
    global index, chunks, metadata
    chunks = []
    metadata = []

    for filename in os.listdir(DOCUMENTS_FOLDER):
        path = os.path.join(DOCUMENTS_FOLDER, filename)
        text = ""

        if filename.endswith(".pdf"):
            reader = PdfReader(path)
            for page in reader.pages:
                text += page.extract_text() or ""
        elif filename.endswith((".txt", ".md")):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

        if not text.strip():
            continue

        # Simple chunking with overlap
        for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk = text[i:i + CHUNK_SIZE]
            chunks.append(chunk)
            metadata.append({"file": filename, "start": i, "chunk_text": chunk})

    if not chunks:
        return "No documents loaded or text extracted."

    # Create embeddings
    embeddings = embedder.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return f"Loaded {len(chunks)} chunks from {len(os.listdir(DOCUMENTS_FOLDER))} files."

def search(question):
    if index is None:
        return "Please load documents first.", ""

    # Embed question
    q_embedding = embedder.encode([question])[0].astype('float32')

    # Search top K
    distances, indices = index.search(np.array([q_embedding]), TOP_K)

    retrieved_chunks = []
    retrieved_display = []

    for idx, dist in zip(indices[0], distances[0]):
        if idx == -1:
            continue
        chunk_info = metadata[idx]
        retrieved_chunks.append(chunk_info["chunk_text"])
        retrieved_display.append(f"**From {chunk_info['file']}** (chunk starting at {chunk_info['start']}):\n{chunk_info['chunk_text'][:300]}...")

    return "\n\n".join(retrieved_chunks), "\n\n---\n\n".join(retrieved_display)

def answer(question):
    context, sources = search(question)

    # Stronger system prompt: forces concise answers + citations
    prompt = f"""You are a helpful assistant answering questions about the AI Engineering book.
Use ONLY the provided context to answer. Be concise, clear, and accurate.
If the information is not in the context, say exactly: "I don't have enough information from the documents."

Always cite the source file and relevant part when possible.

Context:
{context}

Question: {question}

Answer (keep short, use bullet points if helpful):"""

    try:
        response = ollama.generate(
            model='tinyllama',  # change to 'phi3.5' if you prefer better quality
            prompt=prompt
        )
        full_answer = response['response'].strip()
        return full_answer, sources
    except Exception as e:
        return f"Error generating answer: {str(e)}", sources

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Personal Knowledge Base Q&A (Local RAG)")
    gr.Markdown("Drop PDFs/text files in the 'documents' folder, then click Load.")

    load_btn = gr.Button("Load Documents")
    status = gr.Textbox(label="Status", interactive=False)

    question = gr.Textbox(label="Your Question")
    ask_btn = gr.Button("Ask")

    output = gr.Textbox(label="Answer", lines=8)
    sources_box = gr.Markdown(label="Sources (what the AI actually used)")

    load_btn.click(load_documents, outputs=status)
    ask_btn.click(answer, inputs=question, outputs=[output, sources_box])

demo.launch()