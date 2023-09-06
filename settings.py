################## custom css #########################
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

# css examples how to from the docs
# with gr.Blocks(css=css) as demo:
#     box1 = gr.Textbox(value="Good Job", elem_classes="feedback")
#     box2 = gr.Textbox(value="Failure", elem_id="warning", elem_classes="feedback")