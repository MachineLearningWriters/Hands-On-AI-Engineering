import gradio as gr
from pypdf import PdfReader
import os
import ollama
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd

# === SETTINGS ===
DOCUMENTS_FOLDER = "documents"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3
MODEL = 'tinyllama'  # or 'phi3.5'
TEST_SET = "test_set.csv"

embedder = SentenceTransformer('all-MiniLM-L6-v2')

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

    return f"Loaded {len(chunks)} chunks."

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

    prompt = f"""Use ONLY the context below to answer. Be concise.
If not in context, say exactly: "I don't have enough information from the documents."

Context:
{context}

Question: {question}

Answer:"""

    response = ollama.generate(model=MODEL, prompt=prompt)
    return response['response'].strip(), sources

def judge_answer(question, answer, expected):
    prompt = f"""You are an impartial judge.
Question: {question}
AI Answer: {answer}
Expected Behavior: {expected}

Score 1-5:
1. Faithfulness (no hallucination)
2. Relevance (direct answer)
3. Abstention (says "I don't know" if no info)
4. Overall Quality

Output only:
Score: X/5
Reason: [short]"""

    response = ollama.generate(model=MODEL, prompt=prompt)
    return response['response'].strip()

def run_evaluation():
    try:
        test_df = pd.read_csv(TEST_SET)
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
    return results_df.to_markdown(index=False), "Evaluation complete!"

with gr.Blocks() as demo:
    gr.Markdown("# Project 3: RAG Evaluated – Tested & Scored RAG")
    gr.Markdown("Same as Project 2 + automatic evaluation on fixed test questions.")

    with gr.Tab("Chat"):
        load_btn = gr.Button("Load Documents")
        status = gr.Textbox(label="Status")

        question = gr.Textbox(label="Your Question")
        ask_btn = gr.Button("Ask")

        output = gr.Textbox(label="Answer", lines=8)
        sources_box = gr.Markdown(label="Sources")

        load_btn.click(load_documents, outputs=status)
        ask_btn.click(answer, inputs=question, outputs=[output, sources_box])

    with gr.Tab("Evaluation"):
        gr.Markdown("Run offline test on test_set.csv")
        eval_btn = gr.Button("Run Evaluation")
        eval_output = gr.Markdown(label="Evaluation Results")
        eval_status = gr.Textbox(label="Status")

        def run_eval_ui():
            df_md, msg = run_evaluation()
            return df_md, msg

        eval_btn.click(run_eval_ui, outputs=[eval_output, eval_status])

demo.launch(server_name="127.0.0.1", server_port=7862)
