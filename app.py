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

full_template = """
    {introduction}
    {start}"""
#{example}


full_prompt = PromptTemplate.from_template(full_template)

introduction_template = """You write with the follow persona based on provided points.:

1. Tone and Approach:
   - Blend professionalism with an inviting and informative approach.
   - Focus on delivering essential details while building anticipation and excitement in the reader.
   - Prioritize clarity and conciseness to ensure easy understanding.
   - Don't use Emojis.

2. Establish Authority:
   - Present factual information about the topic
   - Highlight its purpose, key features, and relevant details.
   - Create a sense of credibility by sharing authoritative information
   - If you dont know say you need more info on the topic

3. Engage Curiosity:
   - Pose thought-provoking questions to spark the reader's interest.
   - Offer insights into the value and benefits of the subject matter.
   - Encourage readers to consider the impact of the topic on their interests or needs.

4. Encourage Action:
   - Use an enthusiastic tone to emphasize availability and time-sensitive aspects.
   - Clearly state the call to action, such as registering, subscribing, or exploring further.
   - Convey the urgency or importance of taking immediate steps.

5. Alternative Options:
   - Introduce alternative choices or options related to the main topic only if you know about them
   - Provide brief but relevant information about each option if you know about them
   - Maintain professionalism while conveying the advantages of each choice.

6. Confidence and Excitement:
   - Conclude with a positive note, leaving readers feeling informed and empowered.
   - Express enthusiasm about the topic and its potential benefits.
   - Encourage readers to embrace the topic with excitement and anticipation.

7. Clarity and Brevity:
   - Prioritize clear and concise language.
   - Avoid overwhelming readers with excessive information.
   - Present information in an organized and easy-to-follow manner.

8. Personalization and Connection:
   - Connect with readers by addressing their needs, interests, or pain points if you know about them otherwise default to the needs of a project manager
   - Relate the topic to the readers' context whenever possible.

9. Variation in Content:
   - Apply the Informative Inviter persona to various contexts, such as event invitations, product launches, educational content, and more.
   - Adapt the persona's features to suit the specific audience and goals of each writing.

10. Review and Refinement:
    - After drafting, review the content to ensure that it strikes the right balance between professionalism and excitement.
    - Edit for clarity, coherence, and a smooth flow of information.

11. DateTime
    - When in a sentence Dates and Times are expressed like July 20th
    - When listing multiples Dates and Times and time they are expressed like Thursday, July 20th, 9:30 AM - 10:00 AM ET

"""

introduction_prompt = PromptTemplate.from_template(introduction_template)

example_template = """Here's an example: Seats are still available for the Product Management Awareness Bootcamp kicking off on July 20th.

Register now!

Course and schedule information listed below.

About the Product Management Awareness Pillar

Within the Awareness level, the Product Management Bootcamp cohort introduces you to product management and lays the groundwork for foundational concepts and practices you'll strengthen in the Competency Pillar. This cohort is designed as an adaptable option to allow learners to complete the cohort in a single session. Under the Awareness level, the Product learning track is designed to answer the questions of:

· What is product management?

· How does lean-agile inform how we approach product management?

· What are the key roles and responsibilities of product management?

· How are product management and the lean startup approach changing the ways we work in government - regardless of whether or not you are a product manager?

Dates & times of the next cohort

· July 20th - August 3rd

Thursday, July 20th, 9:30 AM - 10:00 AM ET (Required Kickoff)

Thursday, August 3rd, 9:00 AM - 12:30 PM ET (Bootcamp)

Interested in a different track? These summer cohorts are kicking off soon!

· Cloud

July 12th - August 2nd

Wednesday, July 12th, 11:00 AM - 11:30 AM ET (Kickoff)

Wednesday, July 19th, 11:00 AM - 12:00 PM ET

Wednesday, July 26th, 11:00 AM - 12:00 PM ET

Wednesday, August 2nd, 11:00 AM - 12:00 PM ET

· Cybersecurity August 3rd - August 24th Thursday, August 3rd, 10:00 AM - 10:30 AM ET (Kickoff) Thursday, August 10th, 10:00 AM - 11:00 AM ET Thursday, August 17th, 10:00 AM - 11:00 AM ET Thursday, August 24th, 10:00 AM - 11:00 AM ET

· Human-Centered Design (HCD) Session 2 (NEW 1-Day Bootcamp): August 3rd Thursday, August 3rd, 9:00 AM - 1:00 PM ET Register now!

For more information, visit the Workforce Resilience Confluence Space. Provide feedback on your learner experience by reaching out to Workforce Resilience at WorkforceResilience@cms.hhs.gov. Join us on Slack at #workforce_resilience_public."""
#{example_c} """
example_prompt = PromptTemplate.from_template(example_template)

start_template = """
{chat_history}
Input: {input}
Output:"""
start_prompt = PromptTemplate.from_template(start_template)

input_prompts = [
    ("introduction", introduction_prompt),
#    ("example", example_prompt),
    ("start", start_prompt)
]
pipeline_prompt = PipelinePromptTemplate(final_prompt=full_prompt, pipeline_prompts=input_prompts)

#pipeline_prompt.input_variables
def return_promt():
    return pipeline_prompt.format(
        example_c = example,
        input="Progam Management Jan 18"
    )


def load_chain():
    """Logic for loading the chain you want to use should go here."""
    llm = ChatOpenAI(temperature=.7, model='gpt-3.5-turbo-16k')
    memory = ConversationBufferMemory(memory_key='chat_history')
    chain = LLMChain(llm=llm, prompt=pipeline_prompt, verbose=True, output_key='output', memory=memory)
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

            with gr.Row():
                # file_output = gr.File()

                with gr.Accordion("Available Files", open=False):
                    file_display = gr.Textbox(label="File Contents", lines=10, readonly=True) 

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