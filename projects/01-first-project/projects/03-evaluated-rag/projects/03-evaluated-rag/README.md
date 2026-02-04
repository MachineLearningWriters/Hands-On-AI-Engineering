# Project 3: Evaluated RAG with Offline Test Set

**What it does**  
A local RAG question-answering system over your documents **+ offline evaluation** using a Golden Dataset, rubrics, LLM-as-a-judge, and regression testing — exactly as taught in Chapter 10 of the AI Engineering book.

**Features**  
- Chat interface (Gradio) for asking questions  
- Load PDFs/text from `documents/` folder  
- Grounded answers with source chunks displayed  
- Evaluation tab: runs fixed test set → scores faithfulness, relevance, abstention, quality  
- Results saved to `evaluation_results.csv` for analysis

**What you'll learn**  
- Building reliable RAG (chunking, embeddings, FAISS retrieval, grounding)  
- Creating & using a Golden Dataset (test_set.csv)  
- LLM-as-a-Judge for automated scoring  
- Regression testing: detect quality drops after changes  
- Integrating evaluation into a live app (Gradio tab)

**Requirements**  
- Python 3.10+  
- Ollama with model pulled (e.g. `ollama pull tinyllama` or `phi3.5`)  
- `pip install -r requirements.txt`

**How to run locally**  
1. Put PDFs/text in `documents/` folder  
2. Run: python app.py
3. Open http://127.0.0.1:7860  
4. Chat tab: ask questions  
5. Evaluation tab: click "Run Evaluation" → see scores table

**Evaluation results**  
Saved to `evaluation_results.csv` after each run — open in Excel/Notepad.

**Live demo**  
(Coming soon — hosted on Hugging Face Spaces)

**License**  
MIT — free to use/modify.
