# Machine Learning Writers Projects

This repository contains a series of hands-on projects inspired by my **AI Engineering** book.  
Each project teaches practical skills for building reliable AI systems using large language models (LLMs) — **completely locally**, **free of charge**, and **without needing internet access** after initial setup.

## Why these projects?

- Start simple → gradually increase in complexity  
- Focus on real-world reliability: prompting, RAG, evaluation, memory, tools, monitoring  
- 100% offline after setup (Ollama + open models)  
- Excellent for beginners and intermediate learners who want to go beyond chat interfaces

## Quick Start

1. Install **Ollama** → https://ollama.com  
2. Pull a strong local model (recommended):ollama pull phi3.5, (or try `tinyllama` if your computer is slower)
3. Clone this repo: git clone https://github.com/flashyiyke/machine-learning-n-writers.gitcd machine-learning-n-writers
4. Go to any project folder and follow its `README.md`

## Project List

- **Project 1: Simple Book Companion Chat**  

- **Project 2: Personal Knowledge Base Q&A (Basic Local RAG)**  

- **Project 3: Evaluated RAG with Offline Test Set**  

- **Project 4: Reliable Conversational Agent with Memory & Tools**  

- **Project 5: Personal Summarizer with Quality Dashboard**  

- **Project 6: Chapter Compass – Chapter Explorer & Quiz Generator**

## Recommended Learning Path

Start with Project 1 → move sequentially.  
Each project builds new concepts while reusing familiar tools.

## Tools & Setup Notes

- **Core stack**: Python 3.10+, Ollama, Gradio, sentence-transformers, faiss-cpu, pypdf, pandas  
- Install per project with `pip install -r requirements.txt` (when provided)  
- All projects run fully offline after downloading models

## License

MIT – free to use, modify, and share.

Questions or suggestions → open an issue.

Happy building! 
