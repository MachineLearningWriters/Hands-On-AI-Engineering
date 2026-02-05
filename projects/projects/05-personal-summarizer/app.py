import gradio as gr
from pypdf import PdfReader
import os
import ollama
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

# ────────────────────────────────────────────────
# SETTINGS ─ DO NOT CHANGE THESE
# ────────────────────────────────────────────────
LOG_FILE = "logs/summaries_log.csv"
JUDGE_MODEL = 'phi3.5'           # ← change to 'tinyllama' if phi3.5 is too slow
DOCUMENTS_FOLDER = "documents"

# Create logs folder automatically
os.makedirs("logs", exist_ok=True)

# Create log file if it doesn't exist
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=[
        'timestamp', 'filename', 'summary_length', 'summary_style',
        'summary', 'faithfulness', 'completeness', 'conciseness', 'overall'
    ]).to_csv(LOG_FILE, index=False)

# Load embedding model for scoring
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def read_document(file_path):
    """Read text from PDF or text file."""
    if file_path.endswith(".pdf"):
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text.strip()
        except Exception as e:
            return f"PDF reading error: {str(e)}"
    elif file_path.endswith((".txt", ".md")):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            return f"Text reading error: {str(e)}"
    return "Unsupported file type. Please upload PDF, TXT or MD."

def generate_summary(text, length="medium", style="paragraph"):
    """Generate summary with chosen length and style."""
    length_desc = {
        "short": "2-4 sentences",
        "medium": "5-8 sentences",
        "detailed": "10-15 sentences or bullet points"
    }
    style_desc = {
        "paragraph": "in clear paragraphs",
        "bullets": "in bullet points",
        "executive": "executive style – key takeaways first",
        "technical": "technical style – focus on concepts and terms"
    }

    prompt = f"""Summarize the following text from the AI Engineering book.
Keep the summary {length_desc.get(length, "medium")} long.
Present it {style_desc.get(style, "in paragraphs")}.
Be accurate, objective, faithful to the original. No hallucinations or added information.

Text:
{text[:12000]}

Summary:"""

    try:
        response = ollama.generate(model=JUDGE_MODEL, prompt=prompt)
        return response['response'].strip()
    except Exception as e:
        return f"Summary generation error: {str(e)}"

def evaluate_summary(original, summary):
    """Score summary quality using similarity + heuristics."""
    if not summary or "error" in summary.lower():
        return {"faithfulness": 0, "completeness": 0, "conciseness": 0, "overall": 0}

    orig_emb = embedder.encode(original)
    sum_emb = embedder.encode(summary)
    faithfulness = util.cos_sim(orig_emb, sum_emb)[0][0].item() * 5

    completeness = min(5, (len(summary.split()) / max(1, len(original.split()) / 15)) * 5)

    ideal_ratio = 0.15
    ratio = len(summary) / max(1, len(original))
    conciseness = 5 - abs(ratio - ideal_ratio) * 30
    conciseness = max(1, min(5, conciseness))

    overall = (faithfulness + completeness + conciseness) / 3

    return {
        "faithfulness": round(faithfulness, 1),
        "completeness": round(completeness, 1),
        "conciseness": round(conciseness, 1),
        "overall": round(overall, 1)
    }

def summarize_and_evaluate(file, length, style):
    """Main function: read file → summarize → evaluate → log."""
    if not file:
        return "Please upload a file first.", "", "", ""

    file_path = file.name
    text = read_document(file_path)

    if "error" in text.lower():
        return text, "", "", ""

    summary = generate_summary(text, length, style)
    scores = evaluate_summary(text, summary)

    # Log the result
    log_row = pd.DataFrame([{
        'timestamp': datetime.now().isoformat(),
        'filename': os.path.basename(file_path),
        'summary_length': length,
        'summary_style': style,
        'summary': summary,
        'faithfulness': scores['faithfulness'],
        'completeness': scores['completeness'],
        'conciseness': scores['conciseness'],
        'overall': scores['overall']
    }])
    log_row.to_csv(LOG_FILE, mode='a', header=False, index=False)

    return (
        summary,
        f"Faithfulness: {scores['faithfulness']}/5\nCompleteness: {scores['completeness']}/5\nConciseness: {scores['conciseness']}/5\nOverall: {scores['overall']}/5",
        text[:400] + "..." if len(text) > 400 else text,
        f"Summary logged at {datetime.now().strftime('%H:%M:%S')}"
    )

def load_dashboard():
    """Load summary history and stats for dashboard."""
    try:
        df = pd.read_csv(LOG_FILE)
        if df.empty:
            return "No summaries logged yet.", pd.DataFrame()
        stats = {
            "Total summaries": len(df),
            "Average overall score": round(df['overall'].mean(), 2),
            "Best faithfulness": round(df['faithfulness'].max(), 2),
            "Worst completeness": round(df['completeness'].min(), 2)
        }
        return pd.DataFrame([stats]), df.tail(10)
    except:
        return "No log file found yet.", pd.DataFrame()

# ────────────────────────────────────────────────
# GRADIO INTERFACE
# ────────────────────────────────────────────────
with gr.Blocks() as demo:
    gr.Markdown("# Personal Summarizer – Project 5")
    gr.Markdown("Upload ONE file → choose length & style → summarize → see quality scores → check history in Dashboard.")

    with gr.Row():
        file_input = gr.File(label="Upload ONE PDF or Text file")
        length_dropdown = gr.Dropdown(["short", "medium", "detailed"], value="medium", label="Summary Length")
        style_dropdown = gr.Dropdown(["paragraph", "bullets", "executive", "technical"], value="paragraph", label="Style")

    summarize_btn = gr.Button("Generate Summary & Evaluate")

    summary_output = gr.Textbox(label="Generated Summary", lines=10)
    scores_output = gr.Textbox(label="Quality Scores", lines=5)
    preview = gr.Textbox(label="Document Preview (first 400 chars)", lines=5)
    status = gr.Textbox(label="Status", interactive=False)

    with gr.Tab("Dashboard"):
        gr.Markdown("History of summaries and scores")
        refresh_btn = gr.Button("Refresh Dashboard")
        stats_table = gr.Dataframe(label="Summary Stats")
        history_table = gr.Dataframe(label="Recent Summaries")

        refresh_btn.click(load_dashboard, outputs=[stats_table, history_table])

    summarize_btn.click(
        summarize_and_evaluate,
        inputs=[file_input, length_dropdown, style_dropdown],
        outputs=[summary_output, scores_output, preview, status]
    )

# ────────────────────────────────────────────────
# START THE APP
# ────────────────────────────────────────────────
demo.launch(server_name="127.0.0.1", server_port=7860)