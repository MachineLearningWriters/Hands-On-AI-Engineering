import gradio as gr
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pypdf import PdfReader
import os
import ollama
import pandas as pd
from datetime import datetime
import numexpr
import re

# ====================
# CONFIGURATION
# ====================
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
DOCUMENTS_FOLDER = "documents"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3
MAX_HISTORY = 5
LLM_MODEL = 'phi3.5'  # Better tool following & reasoning

# Load embedding model
embedder = SentenceTransformer(EMBEDDING_MODEL)

# Global vector store
index = None
chunks = []
metadata = []

# Conversation history
history = []

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

# ====================
# TOOLS
# ====================
def calculate(expression):
    try:
        expr = re.sub(r'[^\d\+\-\*/xX(). ]', '', expression).replace('x', '*').replace('X', '*')
        result = numexpr.evaluate(expr)
        return str(result)
    except:
        return "Calculation error."

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S WAT")

TOOLS = {
    "calculate": calculate,
    "get_current_time": get_current_time
}

# ====================
# AGENT LOGIC
# ====================
def agent(question):
    global history

    # Quick math safety net
    math_match = re.search(r'\d+\s*[\+\-\*/xX]\s*\d+', question)
    if math_match:
        expr = math_match.group(0)
        result = calculate(expr)
        if "error" not in result.lower():
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": f"Calculation: {result}"})
            return f"Calculation: {result}", ""

    history.append({"role": "user", "content": question})
    if len(history) > MAX_HISTORY * 2:
        history = history[-MAX_HISTORY * 2:]

    history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])

    # Strict tool decision
    tool_prompt = f"""You MUST reply with EXACTLY one of these, nothing else:

TOOL: calculate
INPUT: the exact math expression (e.g. 15 * 23)

TOOL: get_current_time
INPUT: none

NO_TOOL

Rules:
- ALWAYS use calculate for ANY math, multiplication, addition, numbers together
- Use get_current_time for time/date questions
- Do NOT calculate yourself
- Do NOT explain or add text

History:
{history_text}

Question: {question}"""

    tool_decision = ollama.generate(model=LLM_MODEL, prompt=tool_prompt)['response'].strip()

    tool_msg = ""
    if tool_decision.startswith("TOOL:"):
        try:
            lines = tool_decision.splitlines()
            tool_line = lines[0].strip()
            tool_name = tool_line.split("TOOL: ")[1].strip()
            input_line = lines[1].strip() if len(lines) > 1 else ""
            input_text = input_line.split("INPUT: ")[1].strip() if "INPUT: " in input_line else ""

            if tool_name in TOOLS:
                tool_result = TOOLS[tool_name](input_text)
                tool_msg = f"Tool {tool_name} result: {tool_result}"
                history.append({"role": "assistant", "content": tool_msg})
        except:
            tool_msg = "Tool call failed."

    context, sources = search(question)

    prompt = f"""You are a reliable assistant for the AI Engineering book.
Use ONLY context, history, and tool results.
Be concise, accurate. Cite sources when possible.
If no info, say: "I don't have enough information."

History:
{history_text}

Tool result: {tool_msg}

Context:
{context}

Question: {question}

Answer (short, cite file/chunk when relevant):"""

    try:
        response = ollama.generate(model=LLM_MODEL, prompt=prompt)
        final_answer = response['response'].strip()
        history.append({"role": "assistant", "content": final_answer})
        return final_answer, sources
    except Exception as e:
        return f"Error: {str(e)}", sources

# ====================
# EVALUATION
# ====================
def judge_answer(question, answer_text, expected_behavior, model=LLM_MODEL):
    prompt = f"""You are an impartial judge.
Question: {question}
AI Answer: {answer_text}
Expected: {expected_behavior}

Score (1–5):
1. Faithfulness
2. Relevance
3. Abstention
4. Overall Quality

Output only:
Score: X/5
Reason: [short]"""

    try:
        response = ollama.generate(model=model, prompt=prompt)
        return response['response'].strip()
    except:
        return "Judge error"

def run_evaluation():
    try:
        test_df = pd.read_csv("test_set.csv")
    except FileNotFoundError:
        return None, "test_set.csv not found."

    results = []

    for idx, row in test_df.iterrows():
        real_answer, sources = agent(row['question'])
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
    gr.Markdown("# AI Engineering Book Companion (Project 4)")
    gr.Markdown("Chat with memory & tools. Evaluation: run test scores.")

    with gr.Tab("Chat"):
        load_btn = gr.Button("Load Documents")
        status = gr.Textbox(label="Status", interactive=False)

        question = gr.Textbox(label="Your Question")
        ask_btn = gr.Button("Ask")

        output = gr.Textbox(label="Answer", lines=8)
        sources_box = gr.Markdown(label="Sources")

        load_btn.click(load_documents, outputs=status)
        ask_btn.click(agent, inputs=question, outputs=[output, sources_box])

    with gr.Tab("Evaluation"):
        gr.Markdown("Run offline test on test_set.csv")
        eval_btn = gr.Button("Run Evaluation")
        eval_output = gr.Dataframe(label="Results")
        eval_status = gr.Textbox(label="Status", interactive=False)

        def run_eval_ui():
            df, msg = run_evaluation()
            return df, msg

        eval_btn.click(run_eval_ui, outputs=[eval_output, eval_status])

demo.launch()