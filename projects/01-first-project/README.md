# Project 1: Book Chat – Simple LLM Q&A

**What it does**  
A basic local chatbot that answers questions about AI Engineering book topics using Ollama + tinyllama (or phi3.5).  
No memory, no documents — just pure LLM Q&A to show the foundation.

**How to run**  
1. Install Ollama & pull model: `ollama pull tinyllama`  
2. `pip install gradio ollama`  
3. `python app.py`  
4. Open http://127.0.0.1:7860

**Demo**  
Ask: “What is the RAG Triad?”  
It should answer from book concepts.  
Ask something off-topic like “What is my favorite color?” → it refuses or says it doesn’t know.
