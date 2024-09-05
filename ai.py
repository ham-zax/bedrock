import gradio as gr
import boto3
import json
import asyncio

primary_region = "eu-west-3"
bedrock_runtime = boto3.client("bedrock-runtime", region_name=primary_region)
inferenceProfileId = 'eu.anthropic.claude-3-5-sonnet-20240620-v1:0'

def chat_with_ai(message, history, system_prompt, temperature, topP):
    try:
        response = bedrock_runtime.converse(
            modelId=inferenceProfileId,
            system=[{"text": system_prompt}],
            messages=[{
                "role": "user",
                "content": [{"text": message}]
            }],
            inferenceConfig={
                "temperature": temperature,
                "topP": topP,
                "maxTokens": 4096
            }
        )
        
        ai_response = response['output']['message']['content'][0]['text']
        return ai_response
    except Exception as e:
        return f"An error occurred: {str(e)}"

def stream_chat(message, history, system_prompt, temperature, topP):
    full_response = chat_with_ai(message, history, system_prompt, temperature, topP)
    words = full_response.split()
    for i in range(0, len(words), 3):
        partial_response = " ".join(words[:i+3])
        yield partial_response

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AWS AI Services Expert Chat")
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=400)
            with gr.Row():
                msg = gr.Textbox(label="Ask about AWS AI services", scale=4)
                enter_button = gr.Button("Enter", scale=1)
            clear = gr.Button("Clear")
        
        with gr.Column(scale=1):
            system_prompt = gr.Textbox(
                label="System Prompt", 
                value="You are an expert on AWS AI services.",
                lines=3
            )
            temperature = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.2,
                step=0.1,
                label="Temperature"
            )
            topP = gr.Slider(
                minimum=0.0,
                maximum=0.999,
                value=0.988,
                step=0.001,
                label="Top P"
            )
    
    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history, system_prompt, temperature, topP):
        user_message = history[-1][0]
        bot_message = ""
        for partial_response in stream_chat(user_message, history, system_prompt, temperature, topP):
            bot_message = partial_response
            history[-1][1] = bot_message
            yield history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [chatbot, system_prompt, temperature, topP], chatbot
    )
    enter_button.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [chatbot, system_prompt, temperature, topP], chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

demo.queue()
demo.launch()