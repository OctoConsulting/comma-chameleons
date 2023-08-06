import random
import time
from langchain import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import transformers
import torch
import numpy as np
import gradio as gr

# additional styling to .chatinterface example
# def yes_man(message, history):
#     if message.endswith("?"):
#         return "Yes"
#     else:
#         return "Ask me anything!"

# gr.ChatInterface(
#     yes_man,
#     chatbot=gr.Chatbot(height=300),
#     textbox=gr.Textbox(placeholder="Ask me a yes or no question", container=False, scale=7),
#     title="Yes Man",
#     description="Ask Yes Man any question",
#     theme="soft",
#     examples=["Hello", "Am I cool?", "Are tomatoes vegetables?"],
#     cache_examples=True,
#     retry_btn=None,
#     undo_btn="Delete Previous",
#     clear_btn="Clear",
# ).launch()


# tokens slider example
# def echo2(message, history, system_prompt, tokens):
#     response = f"System prompt: {system_prompt}\n Message: {message}."
#     for i in range(min(len(response), int(tokens))):
#         time.sleep(0.05)
#         yield response[: i+1]

# with gr.Blocks() as demo:
#     system_prompt = gr.Textbox("You are helpful AI.", label="System Prompt")
#     slider = gr.Slider(10, 100, render=False)
    
#     gr.ChatInterface(
#         echo2, additional_inputs=[system_prompt, slider]
#     )

# demo.queue().launch()


# langchain example
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


# basic blocks with python event listener example
# def greet(name):
#     return "Hello " + name + "!"

# with gr.Blocks() as demo:
#     name = gr.Textbox(label="Name")
#     output = gr.Textbox(label="Output Box")
#     greet_btn = gr.Button("Greet")
#     greet_btn.click(fn=greet, inputs=name, outputs=output, api_name="greet")
   
# demo.launch()


# flip text block example
# def flip_text(x):
#     return x[::-1]

# def flip_image(x):
#     return np.fliplr(x)

# with gr.Blocks() as demo:
#     gr.Markdown("Flip text or image files using this demo.")
#     with gr.Tab("Flip Text"):
#         text_input = gr.Textbox()
#         text_output = gr.Textbox()
#         text_button = gr.Button("Flip")
#     with gr.Tab("Flip Image"):
#         with gr.Row():
#             image_input = gr.Image()
#             image_output = gr.Image()
#         image_button = gr.Button("Flip")

#     with gr.Accordion("Open for More!"):
#         gr.Markdown("Look at me...")

#     text_button.click(flip_text, inputs=text_input, outputs=text_output)
#     image_button.click(flip_image, inputs=image_input, outputs=image_output)

# demo.launch()

# basic custom css example
# css = """
# #warning {background-color: #FFCCCB} 
# .feedback textarea {font-size: 24px !important}
# """

# with gr.Blocks(css=css) as demo:
#     box1 = gr.Textbox(value="Good Job", elem_classes="feedback")
#     box2 = gr.Textbox(value="Failure", elem_id="warning", elem_classes="feedback")


# blocks demo
blocks = gr.Blocks()

with blocks as demo:
    subject = gr.Textbox(placeholder="subject")
    verb = gr.Radio(["ate", "loved", "hated"])
    object = gr.Textbox(placeholder="object")

    with gr.Row():
        btn = gr.Button("Create sentence.")
        reverse_btn = gr.Button("Reverse sentence.")
        foo_bar_btn = gr.Button("Append foo")
        reverse_then_to_the_server_btn = gr.Button(
            "Reverse sentence and send to server."
        )

    def sentence_maker(w1, w2, w3):
        return f"{w1} {w2} {w3}"

    output1 = gr.Textbox(label="output 1")
    output2 = gr.Textbox(label="verb")
    output3 = gr.Textbox(label="verb reversed")
    output4 = gr.Textbox(label="front end process and then send to backend")

    btn.click(sentence_maker, [subject, verb, object], output1)
    reverse_btn.click(
        None, [subject, verb, object], output2, _js="(s, v, o) => o + ' ' + v + ' ' + s"
    )
    verb.change(lambda x: x, verb, output3, _js="(x) => [...x].reverse().join('')")
    foo_bar_btn.click(None, [], subject, _js="(x) => x + ' foo'")

    reverse_then_to_the_server_btn.click(
        sentence_maker,
        [subject, verb, object],
        output4,
        _js="(s, v, o) => [s, v, o].map(x => [...x].reverse().join(''))",
    )

demo.launch()