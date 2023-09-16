# Gradio UI 1: "Customization and file uploads", "Gradio blocks style"
# https://www.gradio.app/demos
# https://python.langchain.com/docs/integrations/tools/gradio_tools
import gradio as gr

def read_file_from_path(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    return content

# Process an uploaded file and return its content. Used to allow user to upload local files through Gradio.
def process_uploaded_file(file):
    file_path = file.name        
    content = read_file_from_path(file_path)
    return content

# Update the chat history and display a message based on the provided input.
# This method is example of how to play with how the chat history is 
# displayed on the client side. 
def display_text(message, chat_history):
    chat_history.clear()  # Clear the existing chat history
    
    # Provide content based on the specific message received.
    if message == 'How to Amend an Approved Permit, Registration, or Notice using Permits Online':
        content = '<span id="tab-one-content">Select the application that you would like to amend: ...</span>'
    else:
        content = '<span id="tab-two-content">What permit are you applying for?</span>'
    
    bot_message = content
    chat_history.append((None, bot_message))
    return "", chat_history

# Return information based on the type of permit or registration requested.
# This method is example of how to configure Gradio 'examples' whos output differs from
# their displayed message. 
def display_message(button_text):
    # Create detailed messages for each type of action a user can take.
    if button_text == 'Apply for and update the permit, registration, or notice.':
        return '**Press send to ask the ai more about this** Permits Online makes it easy for you ...'
    elif button_text == 'Apply for a Certificate of Label Approval/Exemption (COLA).':
        return '**Press send to ask the ai more about this** COLAs Online makes it easy for you ...'
    elif button_text == 'Your wine, distilled spirit, or beer/malt beverage may require formula approval or laboratory sample analysis':
        return '**Press send to ask the ai more about this** Your wine, distilled spirit, or beer/malt beverage may require ...'

# Make a call to LLM API that sends Gradio input and history as payload, update the history accordingly.
def predict(inp, history):
    

    print("this is the input:" + inp)
    print(history)
    
    # Authentication for IBM Cloud API
    import requests
    API_KEY = '5Oz7Ma1ZCctfI0I9FU1kbZC6YRWm9jRKddtL-8mJYREx'
    token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
    mltoken = token_response.json()["access_token"]
    
    # Set up headers for the API request
    header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}
    
    # Prepare and send the API request for scoring (prediction)
    payload_scoring = {"input_data": [{"values": [history + [inp]]}]}
    response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/ce451240-81cf-41ac-856d-a00153c75bbd/predictions?version=2021-05-01', 
                       json=payload_scoring, 
                       headers={'Authorization': 'Bearer ' + mltoken})
    
    # Parse and retrieve the response
    response_json = response_scoring.json()
    print(response_json)
    
    # Get the predicted output and update the history
    output = response_json.get('predictions', [{}])[0].get('values', [[""]])[0][0]
    history.append((inp, output))
    return "", history

# Custom css for the Gradio. See Gradio docs for how the classes and ids were implemented. 
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

# Gradio blocks style UI
def initialize_ui():  # Initialize the Gradio UI
    """Initialize and launch the Gradio UI."""
    with gr.Blocks(css=css) as block:
        # Displaying header text
        gr.Markdown("<h3><center>TTB.gov watsonx Demo</center></h3>", elem_id="header-subtext")

        with gr.Row():

            with gr.Column():
                # Input textbox for user queries
                with gr.Row():
                    message = gr.Textbox(
                        label="Input",
                        placeholder="Ask anything...",
                        lines=8,
                        elem_id="chat-prompt"
                    ).style(container=False)

                    # Button to upload a file
                    with gr.Row(elem_id="chat-prompt-row"):
                        with gr.Column(scale=0.09, min_width=0):
                            upload_btn = gr.UploadButton("📁 File", file_types=["", ".", ".csv", ".xls", ".xlsx", "text"], file_count="single", elem_id="upload-btn")

                # Button to submit the query
                with gr.Column(scale=0.09, min_width=0):
                    submit = gr.Button(value="➦", variant="secondary").style(full_width=False)

                # Buttons for example queries
                with gr.Column(elem_id="examples-container"):
                    gr.Markdown("<h3><center>≣ Try Asking</center></h3>", elem_id="related-header-subtext")

                    with gr.Row():
                        with gr.Column(scale=0.2, min_width=0):
                            example1_button = gr.Button("Apply for and update the permit, registration, or notice.", elem_classes="tab-button")
                        with gr.Column(scale=0.2, min_width=0):
                            example2_button = gr.Button("Apply for a Certificate of Label Approval/Exemption (COLA).", elem_classes="tab-button")
                        with gr.Column(scale=0.2, min_width=0):
                            example3_button = gr.Button("Your wine, distilled spirit, or beer/malt beverage may require formula approval or laboratory sample analysis", elem_classes="tab-button")

            # Chatbot display section
            with gr.Column(elem_id="chatbot-container"):
                chatbot = gr.Chatbot(height=700, label="watsonx", show_copy_button=True)

                # Additional buttons for quick links or popular searches
                with gr.Column(elem_id="related-links-container"):
                    gr.Markdown("<h3><center>🌩 Quick Search Popular</center></h3>", elem_id="related-header-subtext")
                    tab1_button = gr.Button("How to Amend an Approved Permit, Registration, or Notice using Permits Online", elem_classes="related-links")
                    tab2_button = gr.Button("What to Gather Before you Apply", elem_classes="related-links-bottom")

        # Binding actions to buttons
        tab1_button.click(display_text, inputs=[tab1_button, chatbot], outputs=[chatbot, chatbot])
        tab2_button.click(display_text, inputs=[tab2_button, chatbot], outputs=[chatbot, chatbot])
        example1_button.click(display_message, inputs=[example1_button], outputs=[message])
        example2_button.click(display_message, inputs=[example2_button], outputs=[message])
        example3_button.click(display_message, inputs=[example3_button], outputs=[message])
        submit.click(predict, inputs=[message, chatbot], outputs=[message, chatbot])

        # Upload button event handling. .upload is an abstraction of an event listener when the user clicks the upload_btn.
        upload_btn.upload(process_uploaded_file, inputs=upload_btn, outputs=message)
        
        # Examples of basic Gradio accordions and chatbot 'examples'
        # with gr.Row():
        #     # file_output = gr.File()

        #     with gr.Accordion("Available Files", open=False):
        #         file_display = gr.Textbox(label="File Contents", lines=10, readonly=True) 

        #     gr.Examples(
        #         examples=[
        #             "Hi! How's it going?",
        #             "What should I do tonight?",
        #             "Whats 2 + 2?",
        #         ],
        #         inputs=message,
        #     )

        # Examples of Gradio HTML components
        # gr.HTML(
        #     "<center>Powered by <a href='https://www.ibm.com/docs/en/watsonx-as-a-service?topic=overview-watsonx'>watsonx 🦜️🔗</a></center>"
        # 

    return block  # Return the UI block
    #block.launch()#debug=True) # Return the UI block when it is outside of a method