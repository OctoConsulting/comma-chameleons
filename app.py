import gradio as gr


# https://www.gradio.app/demos
# https://python.langchain.com/docs/integrations/tools/gradio_tools


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
def get_model_reply():
    print("hello")

# def load_chain():
#     """Logic for loading the chain you want to use should go here."""
#     llm = "llamma2" 
#     chain = ConversationChain(llm=llm)
#     return chain

chain = "load_chain"

block = gr.Blocks(css=".gradio-container {background-color: #d3d3d3; }")

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

    submit.click(get_model_reply, inputs=[message], outputs=[chatbot])
    message.submit(get_model_reply, inputs=[message], outputs=[chatbot]) #, [txt, state], [chatbot, state]
    # btn.upload(get_model_reply, [btn, file_output], file_output)

block.launch(debug=True)