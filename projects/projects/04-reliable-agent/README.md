# Project 4: Reliable Conversational Agent with Memory & Tools

**What it does**  
A local conversational AI that:
- Remembers chat history (short-term memory)  
- Uses tools (calculator, current time) when needed  
- Answers questions grounded in your documents (RAG)  
- Shows sources for transparency  
- Has guardrails & safe failure modes  
- Includes offline evaluation tab (from Project 3)

Built as Project 4 for the "Hands-On AI Engineering" book companion.

**Features**
- Chat interface with memory  
- Tool calling (math, date/time)  
- Document-based answers with citations  
- Evaluation dashboard (faithfulness, relevance, abstention scores)  
- 100% local & free (Ollama + FAISS + Gradio)

**Requirements**
- Python 3.10+  
- Ollama with `phi3.5` pulled (`ollama pull phi3.5`)  
- `pip install -r requirements.txt`

**How to run locally**
1. Put PDFs/text files in `documents/` folder  
2. Run: Python app.py
3. Open http://127.0.0.1:7860  
4. Chat tab: ask questions (try math: "15 times 23", time: "what time is it")  
5. Evaluation tab: click "Run Evaluation" → see scores

**Live demo**  
(Coming soon — hosted on Hugging Face Spaces)

**License**  
MIT — free to use/modify.
