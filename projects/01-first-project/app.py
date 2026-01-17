import gradio as gr
import ollama

def chat_with_ai(message, history):
    # This is where we will talk to the local AI
    response = ollama.chat(
        model='phi3.5:mini',  # we can change this later
        messages=[{'role': 'user', 'content': message}]
    )
    return response['message']['content']

demo = gr.ChatInterface(
    fn=chat_with_ai,
    title="Project 1 - Simple Local AI Chat",
    description="Talk to a free AI model running on your computer!"
)

demo.launch()
