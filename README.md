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
Basic local chatbot that answers questions about the book using Ollama + Gradio.  
Key skills: prompt engineering, system prompts, basic UI.  
[View folder → projects/01-book-companion-chat](projects/01-book-companion-chat)

- **Project 2: Personal Knowledge Base Q&A (Basic Local RAG)**  
Upload your own documents (PDF/text) → ask questions → get grounded answers with sources.  
Key skills: document loading, chunking, embeddings, FAISS, retrieval, grounding.  
[View folder → projects/02-basic-rag](projects/02-basic-rag)

- **Project 3: Evaluated RAG with Offline Test Set**  
RAG system + offline evaluation using Golden Dataset, LLM-as-a-judge, rubrics, regression testing.  
Key skills: evaluation frameworks, faithfulness/relevance scoring, regression protection.  
[View folder → projects/03-evaluated-rag](projects/03-evaluated-rag)

- **Project 4: Reliable Conversational Agent with Memory & Tools**  
Conversational agent with chat memory, simple tools (calculator, current time), guardrails.  
Key skills: conversation memory, tool calling, structured output, safe failure modes.  
[View folder → projects/04-reliable-agent](projects/04-reliable-agent)

- **Project 5: Personal Summarizer with Quality Dashboard**  
Upload one document → generate styled summary → automatic quality scoring → view history & stats.  
Key skills: summarization prompting, faithfulness/completeness/conciseness evaluation, logging, monitoring dashboard.  
[View folder → projects/05-personal-summarizer](projects/05-personal-summarizer)

- **Project 6: [Title TBD – e.g. Advanced Monitored Multi-Tool Agent]**  
[Description placeholder – will be updated when built]  
Key skills: [placeholder]  
[View folder → projects/06-[folder-name]](projects/06-[folder-name])

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
