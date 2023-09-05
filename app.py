import gradio as gr
#from langchain import HuggingFacePipeline
#from langchain.chat_models import ChatOpenAI
#from langchain.schema import AIMessage, HumanMessage
from langchain.prompts import PromptTemplate
#from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory
#from langchain.utilities import WikipediaAPIWrapper
#from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
#import transformers
#import torch
#import random
#import time
import os
#from typing import Optional, Tuple
#from threading import Lock
import configparser
#import pdb
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models import Model

def get_secrets(cfg_file='key.ini',llm=None,project_id=False):
    config=configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__),cfg_file))
    
    if project_id == True:
        try:
            return config['WATSONX_KEY']['project_id']
        except:
            raise SystemExit("could not find watsonx project id")
    if project_id == False:
        try:
            if llm == 'OPENAI':
                return config['OPENAI_API_KEY']['key']
            elif llm == 'WATSONX':
                return config['WATSONX_KEY']['key']
        except:
            raise SystemExit("could not find key")

os.environ["OPENAI_API_KEY"] = get_secrets(llm='OPENAI')  # Replace with your key


#########Build Vector DB###########



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
    #OPENAI
    #llm = ChatOpenAI(temperature=.7, model='gpt-3.5-turbo-16k')
    #WATSONX
    params = {
        GenParams.MAX_NEW_TOKENS: 100,
        GenParams.MIN_NEW_TOKENS: 1,
        GenParams.DECODING_METHOD: DecodingMethods.SAMPLE,
        GenParams.TEMPERATURE: 0.5,
        GenParams.TOP_K: 50,
        GenParams.TOP_P: 1
    }
    credentials = {
        'url': "https://us-south.ml.cloud.ibm.com",
        'apikey' : get_secrets(llm='WATSONX')
    }

    project_id = get_secrets(project_id=True)
   
    flan_t5_model = Model(
    model_id="google/flan-t5-xxl",
    credentials=credentials,
    project_id=project_id)
    llm = WatsonxLLM(model=flan_t5_model)
    
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
#chat-prompt {box-shadow: 3px 3px #20808D, -1em 0 .4em lightgrey;}
#chat-prompt-row {position: absolute; top: 169px;}
#header-subtext {display: flex;}
#related-header-subtext h3 center {display: flex; color: #20808D;}
#upload-btn {width: max-content; font-size: 14px; border: none; background: none; box-shadow: none; z-index: 2;}
#tab-one-content {color: black !important; border: none !important;}
#tab-two-content {color: black;}
#tabs-row {}
div > button.related-links {
    background: transparent !important;
    border: none !important;
    border-bottom: solid !important;
    border-radius: 0 !important;
    border-width: thin !important;
    border-color: lightgrey !important;
    justify-content: flex-start !important;
    padding-left: 0 !important;
    box-shadow: none;
}
div > button.related-links-bottom {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    border-width: thin !important;
    border-color: lightgrey !important;
    justify-content: flex-start !important;
    padding-left: 0 !important;
    box-shadow: none;
}
.tab-button {    
    font-weight: bold !important;
    font-size: 10px !important;
    background: none !important;
    border-radius: 4px !important;
    box-shadow: none !important;
    display: flex !important;
    height: min-content !important;
    width: 100% !important;
}
#chatbot-container {}
.message.user {background: #c7eef3 !important; border: #c7eef3; font-weight: bold;}
.message.bot {background: white !important;}
#related-links-container {row-gap: 0px;}
#examples-container {flex-grow: 0 !important;}
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

    def display_text(message, chat_history):
        chat_history.clear()  # Clear the existing chat history

        if message == 'How to Amend an Approved Permit, Registration, or Notice using Permits Online':
            content = '<span id="tab-one-content">Select the application that you would like to amend: If you need to update information about the business entity ONLY, choose "Create Amendment" for the Application for Original Entity. If you need to update information about a particular commodity operation, choose "Create Amendment" for that operation (e.g. Application for Winery Operations). If you need to update both a commodity operation and the associated business entity information, choose "Create Amendment" for the commodity operation record.  You will be given the opportunity to update both the commodity and entity information as part of this process. </span>'
        else:
            content = '<span id="tab-two-content">What permit are you applying for?</span>'
        bot_message = content
        chat_history.append((None, bot_message))
        return "", chat_history

    def display_message(button_text):
        if button_text == 'Apply for and update the permit, registration, or notice.':
            return '**Press send to ask the ai more about this** Permits Online makes it easy for you to apply for and update the permit, registration, or notice you need to operate a TTB-regulated business. Once you have registered to use Permits Online, just follow the instructions and prompts for submitting your application package.  There is no fee to apply. Permits Online makes it easy for you to apply for and update the permit, registration, or notice you need to operate a TTB-regulated business.  Once you have registered to use Permits Online, just follow the instructions and prompts for submitting your application package.  There is no fee to apply.'
        elif button_text == 'Apply for a Certificate of Label Approval/Exemption (COLA).':
            return '**Press send to ask the ai more about this** COLAs Online makes it easy for you to apply for a Certificate of Label Approval/Exemption (COLA) for your product. Once you have registered, just follow the instructions and prompts for submitting your online application. In most cases, once your application is submitted electronically, you\'re done. There are no forms to sign and... there is no fee to apply. COLAs Online makes it easy for you to apply for a Certificate of Label Approval/Exemption (COLA) for your product. Once you have registered, just follow the instructions and prompts for submitting your online application. In most cases, once your application is submitted electronically, you\'re done. There are no forms to sign and... there is no fee to apply.'
        elif button_text == 'Your wine, distilled spirit, or beer/malt beverage may require formula approval or laboratory sample analysis':
            return '**Press send to ask the ai more about this** Your wine, distilled spirit, or beer/malt beverage may require formula approval or laboratory sample analysis before you may apply for a Certificate of Label Approval, or COLA. We require formula approval most commonly when a product has added flavoring or coloring materials. New to Formula Approval? A formula is a complete list of the ingredients used to make an alcohol beverage and a step-by-step description of how it\'s made. In some cases, we may also require laboratory analysis of the product. Your wine, distilled spirit, or beer/malt beverage may require formula approval or laboratory sample analysis before you may apply for a Certificate of Label Approval, or COLA. We require formula approval most commonly when a product has added flavoring or coloring materials. New to Formula Approval? A formula is a complete list of the ingredients used to make an alcohol beverage and a step-by-step description of how it\'s made. In some cases, we may also require laboratory analysis of the product. Your wine, distilled spirit, or beer/malt beverage may require formula approval or laboratory sample analysis before you may apply for a Certificate of Label Approval, or COLA. We require formula approval most commonly when a product has added flavoring or coloring materials. New to Formula Approval? A formula is a complete list of the ingredients used to make an alcohol beverage and a step-by-step description of how it\'s made. In some cases, we may also require laboratory analysis of the product. Your wine, distilled spirit, or beer/malt beverage may require formula approval or laboratory sample analysis before you may apply for a Certificate of Label Approval, or COLA. We require formula approval most commonly when a product has added flavoring or coloring materials. New to Formula Approval? A formula is a complete list of the ingredients used to make an alcohol beverage and a step-by-step description of how it\'s made. In some cases, we may also require laboratory analysis of the product.'

    gr.Markdown("<h3><center>TTB.gov watsonx Demo</center></h3>", elem_id="header-subtext")

    with gr.Row():

        with gr.Column():

            with gr.Row():
                message = gr.Textbox(
                    label="Input",
                    placeholder="Ask anything...",
                    lines=8,
                    elem_id="chat-prompt"
                ).style(container=False)
                
                with gr.Row(elem_id="chat-prompt-row"):
                    with gr.Column(scale=0.09, min_width=0):
                        upload_btn = gr.UploadButton("📁 File", file_types=["", ".", ".csv",".xls",".xlsx", "text"],  file_count="single", elem_id="upload-btn")

            with gr.Column(scale=0.09, min_width=0):  
                submit = gr.Button(value="➦", variant="secondary").style(full_width=False)
            
            with gr.Column(elem_id="examples-container"):
                gr.Markdown("<h3><center>≣ Try Asking</center></h3>", elem_id="related-header-subtext")

                with gr.Row():
                    with gr.Column(scale=0.2, min_width=0):
                        example1_button = gr.Button("Apply for and update the permit, registration, or notice.", elem_classes="tab-button")
                    with gr.Column(scale=0.2, min_width=0):
                        example2_button = gr.Button("Apply for a Certificate of Label Approval/Exemption (COLA).", elem_classes="tab-button")
                    with gr.Column(scale=0.2, min_width=0):
                        example3_button = gr.Button("Your wine, distilled spirit, or beer/malt beverage may require formula approval or laboratory sample analysis", elem_classes="tab-button")

            # with gr.Row():
            #     # file_output = gr.File()

            #     with gr.Accordion("Available Files", open=False):
            #         file_display = gr.Textbox(label="File Contents", lines=10, readonly=True) 

            # gr.Examples(
            #     examples=[
            #         "Hi! How's it going?",
            #         "What should I do tonight?",
            #         "Whats 2 + 2?",
            #     ],
            #     inputs=message,
            # )

            # gr.HTML("Demo application of a LangChain chain.")

        with gr.Column(elem_id="chatbot-container"):

            chatbot = gr.Chatbot(height=700, label="watsonx", show_copy_button=True)

            with gr.Column(elem_id="related-links-container"):
                gr.Markdown("<h3><center>🌩 Quick Search Popular</center></h3>", elem_id="related-header-subtext")
                tab1_button = gr.Button("How to Amend an Approved Permit, Registration, or Notice using Permits Online", elem_classes="related-links")
                tab2_button = gr.Button("What to Gather Before you Apply", elem_classes="related-links-bottom")

    tab1_button.click(display_text, inputs=[tab1_button, chatbot], outputs=[chatbot, chatbot])
    tab2_button.click(display_text, inputs=[tab2_button, chatbot], outputs=[chatbot, chatbot])

    example1_button.click(display_message, inputs=[example1_button], outputs=[message])
    example2_button.click(display_message, inputs=[example2_button], outputs=[message])
    example3_button.click(display_message, inputs=[example3_button], outputs=[message])

    submit.click(predict, inputs=[message, chatbot], outputs=[message, chatbot])
    message.submit(predict, inputs=[message, chatbot], outputs=[message, chatbot]) #, [txt, state], [chatbot, state]

    # .upload is an abstraction of an event listener when the user clicks the upload_btn
    upload_btn.upload(process_uploaded_file, inputs=upload_btn, outputs=message)

    # gr.HTML(
    #     "<center>Powered by <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗</a></center>"
    # )
block.launch()#debug=True)