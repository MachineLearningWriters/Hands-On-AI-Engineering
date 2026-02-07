import gradio as gr
from pypdf import PdfReader
import os
import ollama
import re
import json
from datetime import datetime

# ====================
# SETTINGS
# ====================
CHAPTER_FOLDER = "chapters"
OUTPUT_FOLDER = "outputs"
JUDGE_MODEL = 'phi3.5'  # or 'tinyllama' if slower computer

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def read_chapter(file_path):
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
            return f"Error reading PDF: {str(e)}"
    elif file_path.endswith((".txt", ".md")):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            return f"Error reading text file: {str(e)}"
    return "Unsupported file type."

def extract_outline(text):
    lines = text.splitlines()
    outline = []
    for line in lines:
        line = line.strip()
        if re.match(r'^(Chapter\s+\d+|Section\s+\d+|\d+\.\d+|\d+\.)\s', line, re.I):
            outline.append(line)
        elif re.match(r'^[A-Z][A-Za-z\s]{10,}$', line) and len(line) < 80:
            outline.append(line)
    return "\n".join(outline[:15]) if outline else "No clear outline detected."

def generate_concept_cards(text):
    prompt = f"""Extract 6–8 key concepts from this chapter text.
For each concept:
- Concept name (short)
- 1-sentence explanation
- Why it matters (1 sentence)

Output only JSON array:
[{{"concept": "...", "explanation": "...", "why_matters": "..."}}, ...]

Text:
{text[:8000]}"""

    try:
        response = ollama.generate(model=JUDGE_MODEL, prompt=prompt)
        raw = response['response'].strip()
        if raw.startswith("```json"):
            raw = raw.split("```json")[1].split("```")[0].strip()
        return json.loads(raw)
    except:
        return [{"concept": "Error", "explanation": "Could not generate cards", "why_matters": ""}]

def generate_quiz(text):
    prompt = f"""Create 10 self-test quiz questions from this chapter text.
Mix: 4 multiple choice, 3 short answer, 3 true/false.
For each question:
- question text
- options (if MC)
- correct_answer
- explanation (1-2 sentences)

Output only JSON array:
[{{"type": "mc/short/tf", "question": "...", "options": ["A", "B", ...] or null, "correct": "A" or "short answer text", "explanation": "..."}}, ...]

Text:
{text[:8000]}"""

    try:
        response = ollama.generate(model=JUDGE_MODEL, prompt=prompt)
        raw = response['response'].strip()
        if raw.startswith("```json"):
            raw = raw.split("```json")[1].split("```")[0].strip()
        return json.loads(raw)
    except:
        return [{"type": "error", "question": "Quiz generation failed", "options": null, "correct": "", "explanation": ""}]

def generate_study_pack(file):
    if not file:
        return "Upload a chapter file first.", "", "", ""

    file_path = file.name
    text = read_chapter(file_path)
    if "error" in text.lower():
        return text, "", "", ""

    outline = extract_outline(text)
    cards = generate_concept_cards(text)
    quiz = generate_quiz(text)

    # Format output
    cards_html = "<h3>Key Concept Cards</h3><ul>"
    for card in cards:
        cards_html += f"<li><strong>{card['concept']}</strong><br>{card['explanation']}<br><em>Why it matters:</em> {card['why_matters']}</li>"
    cards_html += "</ul>"

    quiz_html = "<h3>Self-Test Quiz (10 questions)</h3><form>"
    for i, q in enumerate(quiz, 1):
        if q['type'] == "mc":
            quiz_html += f"<p><strong>Q{i}:</strong> {q['question']}</p><ul>"
            for opt in q['options'] or []:
                quiz_html += f"<li>{opt}</li>"
            quiz_html += "</ul>"
        else:
            quiz_html += f"<p><strong>Q{i}:</strong> {q['question']}</p>"
    quiz_html += "</form><p>Answers & explanations hidden – check manually against the chapter.</p>"

    html_report = f"""
    <html>
    <head><title>Chapter Compass – Study Pack</title></head>
    <body>
    <h1>Chapter Compass Study Pack</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    <h2>Outline</h2><pre>{outline}</pre>
    {cards_html}
    {quiz_html}
    </body>
    </html>
    """

    report_path = os.path.join("outputs", f"study-pack-{os.path.basename(file_path)}.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_report)

    return outline, cards_html, quiz_html, f"Study pack saved to: {report_path}"

with gr.Blocks() as demo:
    gr.Markdown("# Chapter Compass – Project 6")
    gr.Markdown("Upload ONE chapter (PDF or text) → get outline, concept cards, quiz, and downloadable study pack.")

    file_input = gr.File(label="Upload ONE chapter file (PDF or TXT)")
    generate_btn = gr.Button("Generate Study Pack")

    outline_output = gr.Textbox(label="Chapter Outline", lines=8)
    cards_output = gr.HTML(label="Key Concept Cards")
    quiz_output = gr.HTML(label="Self-Test Quiz")
    download_status = gr.Textbox(label="Download Status", interactive=False)

    generate_btn.click(
        generate_study_pack,
        inputs=file_input,
        outputs=[outline_output, cards_output, quiz_output, download_status]
    )

demo.launch(server_name="127.0.0.1", server_port=7860)