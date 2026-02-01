# Project 2: Personal Knowledge Base Q&A (Basic Local RAG)

**What it does**  
A 100% local, zero-cost question-answering system over your own documents (PDFs, text, Markdown).  
Ask anything — answers are grounded in your files (no hallucinations when retrieval works well).

**What you'll learn by following along**  
- Loading & parsing PDFs/text files  
- Chunking documents with overlap for better context  
- Creating embeddings with sentence-transformers  
- Storing vectors locally with FAISS (fast similarity search)  
- Retrieval: finding relevant chunks for a question  
- Generation: prompting Ollama to answer only using retrieved content  
- Simple Gradio UI for chatting  
- Displaying sources (so you see exactly what the AI used)

**Requirements**  
- Python 3.10+  
- Ollama with a model pulled (e.g. `ollama pull tinyllama` or `phi3.5`)  
- Packages: `pip install sentence-transformers faiss-cpu pypdf gradio ollama`

**How to run**  
1. Put your PDFs/text files in the `documents/` folder  
2. Run:python app.py
3. Open http://127.0.0.1:7860 in your browser  
4. Click "Load Documents" → ask questions!

**Example questions**  
- "What is the RAG Triad?"  
- "Explain regression testing from the book"  
- "What deployment pattern is best for beginners?"

**Screenshots**  
(Add screenshots later — see below)

**License**  
MIT — feel free to use/modify for your own projects.
