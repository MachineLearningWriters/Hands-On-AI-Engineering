import gradio as gr
import ollama

def answer_question(message, history):
    # Simple system prompt to focus on the book
    system_prompt = "You are a helpful assistant that only answers questions about AI Engineering topics from the book. If the question is not related, say 'I only answer questions about the AI Engineering book.' Keep answers short and clear."

    # Send to Ollama
    response = ollama.chat(
        model='tinyllama',  # change to 'phi3.5' if you want better answers
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': message}
        ]
    )
    return response['message']['content']

demo = gr.ChatInterface(
    fn=answer_question,
    title="Project 1: Book Chat – Simple LLM Q&A",
    description="Ask anything about AI Engineering book topics!",
    examples=[
        "What is the RAG Triad?",
        "Explain LoRA fine-tuning",
        "What is my favorite color?"
    ]
)

demo.launch()
