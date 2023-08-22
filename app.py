import gradio as gr

from langchain import HuggingFacePipeline
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from langchain.prompts import PromptTemplate
from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import transformers
import torch
import random
import time
import os
from typing import Optional, Tuple
from threading import Lock
import configparser

def get_secrets(cfg_file='key.ini'):
    config=configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__),cfg_file))
    try:
        return config['OPENAI_API_KEY']['key']
    except:
        raise SystemExit("could not find key")

os.environ["OPENAI_API_KEY"] = get_secrets()  # Replace with your key


# https://www.gradio.app/demos
# https://python.langchain.com/docs/integrations/tools/gradio_tools

########### Prompt Engineering aka LangChain #######################
#####Return LangChain Promt to bot for testing UI#####

# Define the template for the conversation prompt
template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{chat_history}
Human: {input}
AI:"""

# Create the prompt from the template
prompt = PromptTemplate(
    input_variables=["input", "chat_history"], template= template
)



def load_chain():
    """Logic for loading the chain you want to use should go here."""
    llm = ChatOpenAI(temperature=.7, model='gpt-3.5-turbo-16k')
    memory = ConversationBufferMemory(memory_key='chat_history')
    chain = LLMChain(llm=llm, prompt=prompt, verbose=True, output_key='output', memory=memory)
    return chain




#Old AutoBot
# def get_model_reply(message, chat_history):
#    bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
#    chat_history.append((message, bot_message))
#    time.sleep(1)
#    print(chat_history)
#    return "", chat_history


######################## Gradio ui ##############################
################## basic custom css how to #########################
css = """
#warning {background-color: #FFCCCB} 
.feedback textarea {font-size: 24px !important}
#chat-prompt {box-shadow: 3px 3px red, -1em 0 .4em olive;}
#chat-prompt-row {position: absolute; top: 305px;}
#header-subtext {display: flex;}
#upload-btn {width: max-content; font-size: 14px; border: none; background: none; box-shadow: none; z-index: 2;}
"""
# css examples
# with gr.Blocks(css=css) as demo:
#     box1 = gr.Textbox(value="Good Job", elem_classes="feedback")
#     box2 = gr.Textbox(value="Failure", elem_id="warning", elem_classes="feedback")

block = gr.Blocks(css=css)

with block:

    
    #init chain 
    chain = load_chain()

    ################## BOT #######################
    def predict(inp, history):
        #output = chain.run({'example_c':example,'input':inp})
        output = chain.run({'input':inp})
        history.append((inp, output))
        return "", history

    def read_file_from_path(file_path):
        with open(file_path, 'r') as f:
            content = f.read()
        return content

    def process_uploaded_file(file):
        file_path = file.name        
        content = read_file_from_path(file_path)
        return content

    gr.Markdown("<h3><center>LangChain Demo</center></h3>", elem_id="header-subtext")

    with gr.Row():

        with gr.Column():

            with gr.Row():
                message = gr.Textbox(
                    label="Input",
                    placeholder="What's the answer to life, the universe, and everything?",
                    lines=15,
                    elem_id="chat-prompt"
                ).style(container=False)
                
                with gr.Row(elem_id="chat-prompt-row"):
                    with gr.Column(scale=0.09, min_width=0):
                        upload_btn = gr.UploadButton("📁 File", file_types=["", ".", ".csv",".xls",".xlsx", "text"],  file_count="single", elem_id="upload-btn")

            with gr.Column(scale=0.09, min_width=0):  
                        submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

            # with gr.Row():
            #     # file_output = gr.File()

            #     with gr.Accordion("Available Files", open=False):
            #         file_display = gr.Textbox(label="File Contents", lines=10, readonly=True) 

            gr.Examples(
                examples=[
                    "Hi! How's it going?",
                    "What should I do tonight?",
                    "Whats 2 + 2?",
                ],
                inputs=message,
            )

            # gr.HTML("Demo application of a LangChain chain.")
        
        chatbot = gr.Chatbot(height=700, label="Output")

    submit.click(predict, inputs=[message, chatbot], outputs=[message, chatbot])
    message.submit(predict, inputs=[message, chatbot], outputs=[message, chatbot]) #, [txt, state], [chatbot, state]

    # .upload is an abstraction of an event listener when the user clicks the upload_btn
    upload_btn.upload(process_uploaded_file, inputs=upload_btn, outputs=message)

    # gr.HTML(
    #     "<center>Powered by <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗</a></center>"
    # )
block.launch()#debug=True)