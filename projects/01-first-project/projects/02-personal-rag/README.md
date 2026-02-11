# Project 2: Doc RAG – Document Q&A with Sources

**What it does**  
Upload your own PDF or text files → ask any question → get accurate answers **grounded in your documents** + see exactly which parts (chunks) were used to generate the answer.

This is the first real introduction to **RAG** (Retrieval-Augmented Generation) — the technique that makes AI answers reliable instead of guessing.

**Key skills you learn by following along**
- Loading & parsing PDFs and text files  
- Splitting documents into smart chunks (with overlap)  
- Creating embeddings (text → numbers) using sentence-transformers  
- Storing embeddings in a local vector store (FAISS)  
- Retrieving relevant chunks for a question  
- Prompting the LLM to answer only from retrieved content (no hallucinations)  
- Showing sources so users can verify the answer

**Requirements**
- Python 3.10+  
- Ollama with a model pulled (e.g. `ollama pull tinyllama` or `phi3.5`)  
- `pip install -r requirements.txt`

**How to run locally**
1. Put one or more PDFs/text files in the `documents/` folder  
2. Run:
