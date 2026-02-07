# Project 6: Chapter Compass – Chapter Explorer & Quiz Generator

**What it does**  
Upload one chapter (PDF or text file) → instantly get:  
- Clean chapter outline (headings + subheadings)  
- 6–8 key concept cards (name + explanation + why it matters)  
- 10 self-test quiz questions (multiple choice, short answer, true/false) with hidden answers & explanations  
- Downloadable HTML study pack (open in any browser)

Built as Project 6 for the "Hands-On AI Engineering" book companion.

**Key skills you learn**  
- Reading & parsing PDFs/text files  
- Generating structured output with Ollama (outline, cards, quiz)  
- Creating self-test quizzes with correct answers & explanations  
- Producing downloadable HTML reports  
- Simple, focused single-file processing (no RAG, no multi-document)

**Requirements**  
- Python 3.10+  
- Ollama with `phi3.5` pulled (`ollama pull phi3.5`) or `tinyllama`  
- `pip install -r requirements.txt`

**How to run locally**  
1. Put **one** chapter file in the `chapters/` folder  
2. Run: python app.py
3. Open http://127.0.0.1:7860  
4. Upload the chapter → click "Generate Study Pack"  
5. View outline, cards, quiz → download HTML report from `outputs/`

**Example output**  
- Outline: Chapter headings & subheadings  
- Concept cards: 6–8 focused explanations  
- Quiz: 10 questions + answers (hidden until you check)  
- HTML file: clean, printable study pack

**License**  
MIT – free to use/modify.
