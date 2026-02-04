import gradio as gr
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pypdf import PdfReader
import os
import ollama
import pandas as pd

# ====================
# CONFIGURATION
# ====================
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
DOCUMENTS_FOLDER = "documents"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3

# Load embedding model once
embedder = SentenceTransformer(EMBEDDING_MODEL)

# Global vector store
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

        for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk = text[i:i + CHUNK_SIZE]
            chunks.append(chunk)
            metadata.append({"file": filename, "start": i, "chunk_text": chunk})

    if not chunks:
        return "No documents loaded."

    embeddings = embedder.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

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

    prompt = f"""You are a helpful assistant answering questions strictly based on the AI Engineering book.
Use ONLY the provided context. Be concise, accurate.
If not in context, say: "I don't have enough information from the documents."

Always cite source file and chunk when possible.

Context:
{context}

Question: {question}

Answer (short, bullet points if helpful):"""

    try:
        response = ollama.generate(model='tinyllama', prompt=prompt)
        return response['response'].strip(), sources
    except Exception as e:
        return f"Error: {str(e)}", sources

# ====================
# EVALUATION FUNCTIONS
# ====================
def judge_answer(question, answer_text, expected_behavior, model='tinyllama'):
    prompt = f"""You are an impartial judge.
Question: {question}
AI Answer: {answer_text}
Expected: {expected_behavior}

Score (1–5):
1. Faithfulness (no hallucination)
2. Relevance (direct answer)
3. Abstention (says "I don't know" if no info)
4. Overall Quality

Output only:
Score: X/5
Reason: [1-2 sentences]
"""

    try:
        response = ollama.generate(model=model, prompt=prompt)
        return response['response'].strip()
    except Exception as e:
        return f"Judge error: {str(e)}"

def run_evaluation():
    try:
        test_df = pd.read_csv("test_set.csv")
    except FileNotFoundError:
        return None, "test_set.csv not found."

    results = []

    for idx, row in test_df.iterrows():
        real_answer, sources = answer(row['question'])
        score = judge_answer(row['question'], real_answer, row['expected_behavior'])
        results.append({
            'question': row['question'],
            'category': row['category'],
            'answer': real_answer,
            'sources': sources,
            'score': score
        })

    results_df = pd.DataFrame(results)
    return results_df, "Evaluation complete!"

# ====================
# GRADIO UI
# ====================
with gr.Blocks() as demo:
    gr.Markdown("# AI Engineering Book Companion + Evaluation")
    gr.Markdown("Chat: Ask questions. Evaluation: Run test set scores.")

    with gr.Tab("Chat"):
        load_btn = gr.Button("Load Documents")
        status = gr.Textbox(label="Status", interactive=False)

        question = gr.Textbox(label="Your Question")
        ask_btn = gr.Button("Ask")

        output = gr.Textbox(label="Answer", lines=8)
        sources_box = gr.Markdown(label="Sources")

        load_btn.click(load_documents, outputs=status)
        ask_btn.click(answer, inputs=question, outputs=[output, sources_box])

    with gr.Tab("Evaluation"):
        gr.Markdown("Run offline evaluation on test_set.csv")
        eval_btn = gr.Button("Run Evaluation")
        eval_output = gr.Dataframe(label="Results")
        eval_status = gr.Textbox(label="Status", interactive=False)

        def run_eval_ui():
            df, msg = run_evaluation()
            return df, msg

        eval_btn.click(run_eval_ui, outputs=[eval_output, eval_status])

demo.launch()