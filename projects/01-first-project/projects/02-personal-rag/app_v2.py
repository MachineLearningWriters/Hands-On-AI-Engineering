import gradio as gr
from pypdf import PdfReader
import os
import ollama
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# === SETTINGS ===
DOCUMENTS_FOLDER = "documents"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3
MODEL = 'tinyllama'  # change to 'phi3.5' if you want better answers

# Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

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
            try:
                reader = PdfReader(path)
                for page in reader.pages:
                    text += page.extract_text() or ""
            except Exception as e:
                print(f"PDF error {filename}: {e}")
                continue

        elif filename.endswith((".txt", ".md")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception as e:
                print(f"Text error {filename}: {e}")
                continue

        if not text.strip():
            continue

        # Chunk text
        for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk = text[i:i + CHUNK_SIZE]
            chunks.append(chunk)
            metadata.append({"file": filename, "start": i, "chunk_text": chunk})

    if not chunks:
        return "No documents loaded."

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
        return "", ""

    q_embedding = embedder.encode([question])[0].astype('float32')
    distances, indices = index.search(np.array([q_embedding]), TOP_K)

    retrieved_chunks = []
    retrieved_display = []

    for idx, dist in zip(indices[0], distances[0]):
        if idx == -1:
            continue
        chunk_info = metadata[idx]
        retrieved_chunks.append(chunk_info["chunk_text"])
        display_text = f"**From {chunk_info['file']}** (chunk {chunk_info['start']}):\n{chunk_info['chunk_text'][:300]}..."
        retrieved_display.append(display_text)

    return "\n\n".join(retrieved_chunks), "\n\n---\n\n".join(retrieved_display)

def answer(question):
    context, sources = search(question)

    prompt = f"""You are a helpful assistant answering questions based ONLY on the provided context from documents.
Use ONLY the context below. Be concise and accurate.
If the information is not in the context, say exactly: "I don't have enough information from the documents."

Context:
{context}

Question: {question}

Answer (short, cite source file/chunk when possible):"""

    try:
        response = ollama.generate(model=MODEL, prompt=prompt)
        return response['response'].strip(), sources
    except Exception as e:
        return f"Error: {str(e)}", sources

with gr.Blocks() as demo:
    gr.Markdown("# Project 2: Doc RAG – Document Q&A with Sources")
    gr.Markdown("Upload your documents → ask questions → get answers with exact sources shown.")

    load_btn = gr.Button("Load Documents from 'documents' folder")
    status = gr.Textbox(label="Status", interactive=False)

    question = gr.Textbox(label="Your Question")
    ask_btn = gr.Button("Ask")

    output = gr.Textbox(label="Answer", lines=8)
    sources_box = gr.Markdown(label="Sources Used")

    load_btn.click(load_documents, outputs=status)
    ask_btn.click(answer, inputs=question, outputs=[output, sources_box])

demo.launch(server_name="127.0.0.1", server_port=7861)
