
# Local Imports
from llm_utils import *
from app_utils import initialize_ui
#from settings import css
#from gradio import *


###### Build Vector DB ########
#url= "https://js.langchain.com/docs/modules/memory/"
#docs = web_scrap(url)
#vector_db = create_vector_db(docs)

###### Langchain ########
#chain = load_chain(vector_db)

###### Assistant Api ########

###### initialize a Gradio? ########
block = initialize_ui()
   
block.launch()#debug=True)

# if __name__ == "__main__":
#     main()





# frontend = embedded chat ui in your HTML
# backend = the webhook endpoint running the llms
# 1. The client will embed Watson Assistant Widget in their HTML
# 2. We will configure our Watson Assistants to trigger a webhook 
# 3. Backend API (the python here) in the cloud will listen for webhook
# 4. Respond payload back to frontend: Setup Watson Assistant to use the webhook response in dialog nodes in IBM Assistant Dashboard 
# @app.route('/webhook', methods=['POST'])
def webhook(request_data):
    context = request_data.get('context', {}) # Assistant comes with/sends a 'context' variable
    user_input = request_data.get('input', {}).get('text', '')
    
    # Append the latest user input to chat history
    chat_history = context.get('chat_history', [])
    chat_history.append({'user': user_input}) # send to a db?

    # Call your LLM/Langchain here and get the response, etc.
    llm_response = chain.run(user_input, context)
    chat_history.append({'llm': llm_response})

    # Save/update context/history here (optional)
    # this is where can customize context and interface with llm. on the Assistant Dashboard we can supposedly access this later and do node things in response to the 'context' values
    # From docs: "In addition to maintaining our place in the conversation, the context can also contain action variables that store any other data you want to pass back and forth between your application and the assistant. This can include persistent data you want to maintain throughout the conversation (such as a customer's name or account number), or any other data you want to track (such as the contents of a shopping cart or user preferences)."
    # context['some_new_variable'] = 'some_value'
    context['chat_history'] = chat_history

    return jsonify({
        'context': context,
        'output': {
            'generic': [
                {
                    'response_type': 'text',
                    'text': llm_response
                }
            ]
        },
    })



################ example of Python SDK style from IBM docs (does not go with our Assistant chatbox in same ways but shows another example of context maybe for later) #############
# https://cloud.ibm.com/docs/watson-assistant?topic=watson-assistant-api-client&code=python

# Example 3: Preserves context to maintain state.
# from ibm_watson import AssistantV2
# from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

# # Create Assistant service object.
# authenticator = IAMAuthenticator('{apikey}') # replace with API key
# assistant = AssistantV2(
#     version = '2021-11-27',
#     authenticator = authenticator
# )
# assistant.set_service_url('{url}') # replace with service instance URL
# assistant_id = '{environment_id}' # replace with environment ID

# # Initialize with empty message to start the conversation.
# message_input = {
#     'message_type:': 'text',
#     'text': ''
#     }
# context = {}

# # Initialize with empty message to start the conversation.
# message_input = {
#     'message_type:': 'text',
#     'text': ''
#     }
# context = {}

# # Main input/output loop
# while message_input['text'] != 'quit':

#     # Send message to assistant.
#     result = assistant.message_stateless(
#         assistant_id,
#         input = message_input,
#         context = context
#     ).get_result()

#     context = result['context']

#     # Print responses from actions, if any. Supports only text responses.
#     if result['output']['generic']:
#         for response in result['output']['generic']:
#             if response['response_type'] == 'text':
#                 print(response['text'])

#     # Prompt for the next round of input unless skip_user_input is True.
#     if not result['context']['global']['system'].get('skip_user_input', False):
#         user_input = input('>> ')
#         message_input = {
#             'text': user_input
#         }