import gradio as gr

def my_func(input_text):
    return f"Output: {input_text}"

iface = gr.Interface(
    my_func,
    [gr.inputs.Textbox()],
    [gr.outputs.Textbox(), 
     gr.outputs.HTML()],
    live=True
)

gr.HTML("html", '''
<script>
  window.watsonAssistantChatOptions = {
    integrationID: "5faa5135-d62b-4842-88c8-dbf9eb29ccc0", // The ID of this integration.
    region: "us-south", // The region your integration is hosted in.
    serviceInstanceID: "793c23d6-8494-4caf-99a9-7349fb064711", // The ID of your service instance.
    onLoad: function(instance) { instance.render(); }
  };
  setTimeout(function(){
    const t=document.createElement('script');
    t.src="https://web-chat.global.assistant.watson.appdomain.cloud/versions/" + (window.watsonAssistantChatOptions.clientVersion || 'latest') + "/WatsonAssistantChatEntry.js";
    document.head.appendChild(t);
  });
</script>
''')

iface.launch()