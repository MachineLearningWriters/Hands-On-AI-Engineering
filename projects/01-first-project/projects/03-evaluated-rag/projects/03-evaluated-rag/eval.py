import pandas as pd
import ollama
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pypdf import PdfReader
import os

# ====================
# CONFIGURATION
# ====================
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'      # Fast & good quality
DOCUMENTS_FOLDER = "documents"             # Folder with your PDFs/text files
CHUNK_SIZE = 500                           # Characters per chunk
CHUNK_OVERLAP = 100                        # Overlap between chunks
TOP_K = 3                                  # Retrieve top 3 chunks
JUDGE_MODEL = 'tinyllama'                  # Can change to 'phi3.5' later

# Load embedding model once
embedder = SentenceTransformer(EMBEDDING_MODEL)

# Global vector store variables
index = None
chunks = []
metadata = []

# ====================
# LOAD DOCUMENTS & BUILD INDEX
# ====================
def load_documents():
    global index, chunks, metadata
    chunks = []
    metadata = []

    print("Loading documents...")

    for filename in os.listdir(DOCUMENTS_FOLDER):
        path = os.path.join(DOCUMENTS_FOLDER, filename)
        text = ""

        # Read PDF
        if filename.endswith(".pdf"):
            try:
                reader = PdfReader(path)
                for page in reader.pages:
                    text += page.extract_text() or ""
            except Exception as e:
                print(f"Error reading PDF {filename}: {e}")
                continue

        # Read txt/md
        elif filename.endswith((".txt", ".md")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue

        if not text.strip():
            continue

        # Chunking with overlap
        for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk = text[i:i + CHUNK_SIZE]
            chunks.append(chunk)
            metadata.append({
                "file": filename,
                "start": i,
                "chunk_text": chunk
            })

    if not chunks:
        return "No documents loaded or no text extracted."

    # Embed all chunks
    print("Creating embeddings...")
    embeddings = embedder.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return f"Loaded {len(chunks)} chunks from {len(os.listdir(DOCUMENTS_FOLDER))} files."

# ====================
# RETRIEVAL
# ====================
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
        display_text = (
            f"**From {chunk_info['file']}** (chunk starting at {chunk_info['start']}):\n"
            f"{chunk_info['chunk_text'][:300]}..."
        )
        retrieved_display.append(display_text)

    return "\n\n".join(retrieved_chunks), "\n\n---\n\n".join(retrieved_display)

# ====================
# ANSWER GENERATION
# ====================
def answer(question):
    context, sources = search(question)

    prompt = f"""You are a helpful assistant answering questions strictly based on the AI Engineering book.
Use ONLY the provided context below to answer. Be concise, clear, accurate, and professional.
If the information is not in the context, say exactly: "I don't have enough information from the documents."

Always cite the source file and relevant part when possible (e.g., "From ai-engineering-book.pdf, chunk starting at 1500: ...").

Context:
{context}

Question: {question}

Answer (keep short, use bullet points if helpful):"""

    try:
        response = ollama.generate(
            model='tinyllama',
            prompt=prompt
        )
        full_answer = response['response'].strip()
        return full_answer, sources
    except Exception as e:
        return f"Error generating answer: {str(e)}", sources

# ====================
# LLM-AS-A-JUDGE
# ====================
def judge_answer(question, answer_text, expected_behavior, model='tinyllama'):
    prompt = f"""You are an impartial judge evaluating an AI answer against expected behavior.
Question: {question}
AI Answer: {answer_text}
Expected Behavior: {expected_behavior}

Score the answer on these criteria (1–5):
1. Faithfulness: Does it stick to facts without hallucination?
2. Relevance: Does it answer the question directly?
3. Abstention: If no info, does it say "I don't have enough information"?
4. Overall Quality

Output only:
Score: X/5
Reason: [short explanation, 1-2 sentences]
"""

    try:
        response = ollama.generate(
            model=model,
            prompt=prompt
        )
        return response['response'].strip()
    except Exception as e:
        return f"Judge error: {str(e)}"

# ====================
# RUN FULL EVALUATION
# ====================
def run_evaluation():
    try:
        test_df = pd.read_csv("test_set.csv")
    except FileNotFoundError:
        print("Error: test_set.csv not found in the folder.")
        return

    results = []

    for idx, row in test_df.iterrows():
        print(f"Evaluating question {idx+1}/{len(test_df)}: {row['question']}")
        real_answer, sources = answer(row['question'])
        score = judge_answer(
            row['question'],
            real_answer,
            row['expected_behavior']
        )
        results.append({
            'question': row['question'],
            'category': row['category'],
            'answer': real_answer,
            'sources': sources,
            'score': score
        })

    results_df = pd.DataFrame(results)
    print("\nEvaluation Results:")
    print(results_df)
    results_df.to_csv("evaluation_results.csv", index=False)
    print("\nEvaluation complete. Results saved to evaluation_results.csv")

if __name__ == "__main__":
    print("Loading documents once for evaluation...")
    load_status = load_documents()
    print(load_status)
    print("\nStarting evaluation...")
    run_evaluation()