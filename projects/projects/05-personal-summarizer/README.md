# Project 5: Personal Summarizer

A simple, local tool to:
- Upload one PDF or text file  
- Generate a summary (choose short/medium/detailed length + paragraph/bullets/executive/technical style)  
- Automatically evaluate quality (faithfulness, completeness, conciseness, overall score)  
- View history of all summaries & scores in a dashboard  

Built as Project 5 for the "Hands-On AI Engineering" book companion.

**Features**
- Single-file summarization (no RAG from multiple docs)  
- Style & length control via dropdowns  
- Quality scoring using LLM-as-a-judge + embedding similarity  
- All summaries logged to `logs/summaries_log.csv`  
- Dashboard shows stats & recent summaries  
- 100% local & free (Ollama + Gradio)

**Requirements**
- Python 3.10+  
- Ollama with `phi3.5` pulled (`ollama pull phi3.5`) or `tinyllama`  
- `pip install -r requirements.txt`

**How to run locally**
1. Put ONE PDF or text file in `documents/` folder  
2. Run: python app.py
3. Open http://127.0.0.1:7860  
4. Upload file → choose length/style → click "Generate Summary & Evaluate"  
5. See summary + quality scores  
6. Go to Dashboard tab → Refresh → see history & stats

**Tips**
- Use short/medium length for long documents (faster, less token usage)  
- Switch to 'tinyllama' in code if phi3.5 is slow  
- Logs saved to `logs/summaries_log.csv` — open in Excel

**License**  
MIT — free to use/modify.
