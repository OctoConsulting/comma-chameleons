# Gradio UI 1: "Customization and file uploads", "Gradio blocks style"
# https://www.gradio.app/demos
# https://python.langchain.com/docs/integrations/tools/gradio_tools
import gradio as gr

# Read the contents of a file
def read_file_from_path(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    return content

# Process an uploaded file and return its content. Used to allow user to upload local files through Gradio.
def process_uploaded_file(file):
    file_path = file.name        
    content = read_file_from_path(file_path)
    return content

# Return information based on the type of permit or registration requested.
# This method is example of how to configure Gradio 'examples' whos output differs from
# their displayed message. 
def display_message(button_text):
    # Create detailed messages for each type of action a user can take.
    if button_text == 'Who is Alan Turing?':
        return 'Who is Alan Turing?'
    elif button_text == 'What is Alan Turing known for?':
        return 'What is Alan Turing known for?'
    elif button_text == 'Who is the person behind the Turing Test?':
        return 'Who is the person behind the Turing Test?'

# Custom css for the Gradio. See Gradio docs for how the classes and ids were implemented. 
css = """
#warning {background-color: #FFCCCB} 
.feedback textarea {font-size: 24px !important}
#chat-prompt-row {position: absolute; top: 398px;}
#header-subtext {display: flex;}
#related-header-subtext h3 center {display: flex; color: #F7931E;}
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
def initialize_ui(predict):  # Initialize the Gradio UI
    """Initialize and launch the Gradio UI."""
    with gr.Blocks(css=css) as block:
        # Displaying header text
        gr.Markdown("<h3><center>watsonx RAG Demo</center></h3>", elem_id="header-subtext")

        with gr.Row():

            with gr.Column():
                # Input textbox for user queries
                with gr.Row():
                    message = gr.Textbox(
                        label="Input",
                        placeholder="Ask anything...",
                        lines=20,
                        elem_id="chat-prompt"
                    ).style(container=False)

                    # Button to upload a file
                    with gr.Row(elem_id="chat-prompt-row"):
                        with gr.Column(scale=0.09, min_width=0):
                            upload_btn = gr.UploadButton("📁 File", file_types=["", ".", ".csv", ".xls", ".xlsx", "text"], file_count="single", elem_id="upload-btn")

                # Button to submit the query
                with gr.Column(scale=0.09, min_width=0):
                    submit = gr.Button(value="Click to Submit", variant="secondary").style(full_width=False)

                # Buttons for example queries
                with gr.Column(elem_id="examples-container"):
                    gr.Markdown("<h3><center>≣ Try Asking</center></h3>", elem_id="related-header-subtext")

                    with gr.Row():
                        with gr.Column(scale=0.2, min_width=0):
                            example1_button = gr.Button("Who is Alan Turing?", elem_classes="tab-button")
                        with gr.Column(scale=0.2, min_width=0):
                            example2_button = gr.Button("What is Alan Turing known for?", elem_classes="tab-button")
                        with gr.Column(scale=0.2, min_width=0):
                            example3_button = gr.Button("Who is the person behind the Turing Test?", elem_classes="tab-button")

            # Chatbot display section
            with gr.Column(elem_id="chatbot-container"):
                chatbot = gr.Chatbot(height=500, label="watsonx", show_copy_button=True)

        example1_button.click(display_message, inputs=[example1_button], outputs=[message])
        example2_button.click(display_message, inputs=[example2_button], outputs=[message])
        example3_button.click(display_message, inputs=[example3_button], outputs=[message])
        submit.click(predict, inputs=[message, chatbot], outputs=[message, chatbot])

        # Upload button event handling. .upload is an abstraction of an event listener when the user clicks the upload_btn.
        upload_btn.upload(process_uploaded_file, inputs=upload_btn, outputs=message)
        
    return block  # Return the UI block