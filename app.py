
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




authenticator = IAMAuthenticator('your-api-key')
assistant = AssistantV2(
    version='your-version-date',
    authenticator=authenticator
)
assistant.set_service_url('your-service-url')

def webhook():
    payload = request.json
    text_input = payload['input']['text']
    session_id = payload['session_id']

    response = assistant.message(
        assistant_id='your-assistant-id',
        session_id=session_id,
        input={
            'message_type': 'text',
            'text': text_input
        }
    ).get_result()

    session_history = assistant.get_session(
        assistant_id='your-assistant-id',
        session_id=session_id
    ).get_result()

    # You can pass response and session history to LLM things here.

    return jsonify({
        'response': response,
        'session_history': session_history
    })