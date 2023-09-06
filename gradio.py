# Gradio UI 1
# https://www.gradio.app/demos
# https://python.langchain.com/docs/integrations/tools/gradio_tools
def initialize_ui(chain: LLMChain, prompt: PromptTemplate):
    """Initialize and launch the Gradio UI."""
    block = gr.Blocks(css=css)
    with block:

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