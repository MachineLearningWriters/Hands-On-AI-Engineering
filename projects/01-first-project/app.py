# Project 1: Simple AI Engineering Book Companion Chat
# Proof of Concept for the "Machine Learning n Writers" / AI Engineering book
# Runs 100% locally with Ollama + tinyllama
# Created by Machine Learning Writers
# No memory, no extra features — just reliable book-topic answers

import gradio as gr
import ollama

def chat_with_ai(message, history):
    response = ollama.chat(
        model='tinyllama',
        messages=[
            {
                'role': 'system',
                'content': """You are "AI Engineering Companion", a friendly helper for the book "Machine Learning n Writers" / "Practical AI Engineering" by Ikenna.

Your only job is to help readers understand and apply the book's main topics:
- Prompt engineering
- Retrieval-Augmented Generation (RAG)
- Evaluation, testing, and reliability
- Guardrails and safe failure modes
- Deployment and monitoring of local AI systems
- Offline / zero-cost tools (Ollama, Gradio, etc.)

Always:
- Answer in simple, beginner-friendly language
- Give short examples or small experiments when possible
- Be encouraging: "Great question! This is exactly what Chapter 10 teaches..."
- If the question has nothing to do with the book or AI engineering, politely say:  
  "I'm focused on helping with the AI Engineering book topics. What part of prompting, RAG, evaluation, reliability or deployment would you like to talk about?"

Stay excited about learning AI engineering!"""
            },
            {'role': 'user', 'content': message}
        ]
    )
    return response['message']['content']

demo = gr.ChatInterface(
    fn=chat_with_ai,
    title="AI Engineering Companion – For the Machine Learning n Writers Book",
    description="Ask anything about the book's topics: prompting, RAG, evaluation, reliability, deployment, monitoring, and local tools. Each question is independent.",
    examples=[
        "What is the most important thing in prompt engineering?",
        "How do I build a simple RAG system?",
        "Why is evaluation so hard for language models?",
        "What does Chapter 11 say about monitoring?",
        "My app is slow — what can I try?"
    ]
)

demo.launch()