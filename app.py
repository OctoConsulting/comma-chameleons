import gradio as gr

from langchain import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import transformers
import torch
import random
import time

# https://www.gradio.app/demos
# https://python.langchain.com/docs/integrations/tools/gradio_tools

#####Return LangChain Promt to bot for testing UI#####
def __return_promt():
    example = """Did you enjoy Leadership Awareness with Dr. Renee Carr? Join Renee as she leads you through a 3-week Leadership Competency cohort focused on Level 1 (Self). This will be the last offering for Level 1 (Self) in 2023.

    Register now!

    Course and schedule information listed below.

    About the Leadership Competency Pillar
    The Leadership Competency Pillar Levels 1-3 are designed to equip agile leaders with advanced, practice-oriented skills for effectively leading teams in a Volatile, Uncertain, Complex, and Ambiguous (VUCA) world. Start with the Level 1 (Self) cohort to discover your personal alignment toward leadership. Upon completion of Level 1 (Self), you can immediately advance into Level 2 (Teams) this summer and Level 3 (Organizations) this fall. For each level, you’ll need 8 total hours across three weeks and earn 8 CLPs.

    Summer 2023
    July 11th — July 25th (Level 1 Self) *Last chance to take Level 1 (Self) in 2023!
    Tuesday, July 11th, 11:00 AM — 2:00 PM ET
    Tuesday, July 18th, 11:00 AM — 2:00 PM ET
    Tuesday, July 25th, 11:00 AM — 1:00 PM ET

    Register now!

    Interested in developing a foundational skill set in another learning track? View course and schedule information on Confluence.

    We hope you will join us,
    Workforce Resilience team

    Provide feedback on your learner experience by reaching out to Workforce Resilience at workforceresilience@cms.hhs.gov. Join us on Slack at #workforce_resilience_public.
    """

    full_template = """{introduction}

    {example}

    {start}"""
    full_prompt = PromptTemplate.from_template(full_template)

    introduction_template = """You create marketing copies for an internal
    IT workforce reliance program for a healthcare insurance organization. """
    introduction_prompt = PromptTemplate.from_template(introduction_template)

    example_template = """Here's some examples of how you write:

    {example_c} """
    example_prompt = PromptTemplate.from_template(example_template)

    start_template = """

    Input: {input}
    Output:"""
    start_prompt = PromptTemplate.from_template(start_template)

    input_prompts = [
        ("introduction", introduction_prompt),
        ("example", example_prompt),
        ("start", start_prompt)
    ]
    pipeline_prompt = PipelinePromptTemplate(final_prompt=full_prompt, pipeline_prompts=input_prompts)

    #pipeline_prompt.input_variables

    return pipeline_prompt.format(
        example_c = example,
        input="Progam Management Jan 18"
    )




################### langchain example method FROM THE GRADIO DOCS #######################
# os.envrion["OPENAI_API_KEY"] = "sk-..."  # Replace with your key

# llm = ChatOpenAI(temperature=1.0, model='gpt-3.5-turbo-0613')

# def predict(message, history):
#     history_langchain_format = []
#     for human, ai in history:
#         history_langchain_format.append(HumanMessage(content=human))
#         history_langchain_format.append(AIMessage(content=ai))
#     history_langchain_format.append(HumanMessage(content=message))
#     gpt_response = llm(history_langchain_format)
#     return gpt_response.content

# gr.ChatInterface(predict).launch() 


#################### "Here a similar example, but this time with streaming results as well:" #######################
# def predict(message, history):
#     history_openai_format = []
#     for human, assistant in history:
#         history_openai_format.append({"role": "user", "content": human })
#         history_openai_format.append({"role": "assistant", "content":assistant})
#     history_openai_format.append({"role": "user", "content": message})

#     response = openai.ChatCompletion.create(
#         model='gpt-3.5-turbo',
#         messages= history_openai_format,         
#         temperature=1.0,
#         stream=True
#     )
    
#     partial_message = ""
#     for chunk in response:
#         if len(chunk['choices'][0]['delta']) != 0:
#             partial_message = partial_message + chunk['choices'][0]['delta']['content']
#             yield partial_message 

# gr.ChatInterface(predict).queue().launch() 


################## basic custom css how to #########################
# css = """
# #warning {background-color: #FFCCCB} 
# .feedback textarea {font-size: 24px !important}
# """

# with gr.Blocks(css=css) as demo:
#     box1 = gr.Textbox(value="Good Job", elem_classes="feedback")
#     box2 = gr.Textbox(value="Failure", elem_id="warning", elem_classes="feedback")


######################## Gradio ui ##############################
def get_model_reply(message, chat_history):
    bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
    chat_history.append((message, bot_message))
    time.sleep(1)
    return "", chat_history

# def load_chain():
#     """Logic for loading the chain you want to use should go here."""
#     llm = "llamma2" 
#     chain = ConversationChain(llm=llm)
#     return chain

chain = "load_chain"

#block = gr.Blocks(css=".gradio-container {background-color: white;}")
#using gradio default scheme for now
block = gr.Blocks(css=".gradio-container}")

with block:
    with gr.Row():
        gr.Markdown("<h3><center>LangChain Demo</center></h3>")

    chatbot = gr.Chatbot()

    with gr.Row():
        message = gr.Textbox(
            label="What's your question?",
            placeholder="What's the answer to life, the universe, and everything?",
            lines=1,
        ).style(container=False)

        with gr.Column(scale=0.15, min_width=0):  
            submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

        with gr.Column(scale=0.15, min_width=0):
            btn = gr.UploadButton("📁Upload", file_types=["image", "audio"],  file_count="single")

    with gr.Row():
        with gr.Accordion("Available Files", open=False):
            file_output = gr.File(file_count="multiple", file_types=["image", "audio"], label="Files Available")

    gr.Examples(
        examples=[
            "Hi! How's it going?",
            "What should I do tonight?",
            "Whats 2 + 2?",
        ],
        inputs=message,
    )

    gr.HTML("Demo application of a LangChain chain.")

    gr.HTML(
        "<center>Powered by <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗</a></center>"
    )

    state = gr.State()
    agent_state = gr.State()


    submit.click(get_model_reply, inputs=[message, chatbot], outputs=[message, chatbot])
    message.submit(get_model_reply, inputs=[message, chatbot], outputs=[message, chatbot]) #, [txt, state], [chatbot, state]
    # btn.upload(get_model_reply, [btn, file_output], file_output)

block.launch(debug=True)