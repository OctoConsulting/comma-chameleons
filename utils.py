def load_chain():
    """Logic for loading the chain you want to use should go here."""
    #OPENAI
    #llm = ChatOpenAI(temperature=.7, model='gpt-3.5-turbo-16k')
    #WATSONX
    params = {
        GenParams.MAX_NEW_TOKENS: 100,
        GenParams.MIN_NEW_TOKENS: 1,
        GenParams.DECODING_METHOD: DecodingMethods.SAMPLE,
        GenParams.TEMPERATURE: 0.5,
        GenParams.TOP_K: 50,
        GenParams.TOP_P: 1
    }
    credentials = {
        'url': "https://us-south.ml.cloud.ibm.com",
        'apikey' : get_secrets(llm='WATSONX')
    }

    project_id = get_secrets(project_id=True)
   
    flan_t5_model = Model(
    model_id="google/flan-t5-xxl",
    credentials=credentials,
    project_id=project_id)
    llm = WatsonxLLM(model=flan_t5_model)
    
    memory = ConversationBufferMemory(memory_key='chat_history')
    chain = LLMChain(llm=llm, prompt=prompt, verbose=True, output_key='output', memory=memory)
    return chain

def get_secrets(cfg_file='key.ini',llm=None,project_id=False):
    config=configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__),cfg_file))
    
    if project_id == True:
        try:
            return config['WATSONX_KEY']['project_id']
        except:
            raise SystemExit("could not find watsonx project id")
    if project_id == False:
        try:
            if llm == 'OPENAI':
                return config['OPENAI_API_KEY']['key']
            elif llm == 'WATSONX':
                return config['WATSONX_KEY']['key']
        except:
            raise SystemExit("could not find key")
        
def read_file_from_path(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    return content

def process_uploaded_file(file):
    file_path = file.name        
    content = read_file_from_path(file_path)
    return content

def display_text(message, chat_history):
    chat_history.clear()  # Clear the existing chat history

    if message == 'How to Amend an Approved Permit, Registration, or Notice using Permits Online':
        content = '<span id="tab-one-content">Select the application that you would like to amend: If you need to update information about the business entity ONLY, choose "Create Amendment" for the Application for Original Entity. If you need to update information about a particular commodity operation, choose "Create Amendment" for that operation (e.g. Application for Winery Operations). If you need to update both a commodity operation and the associated business entity information, choose "Create Amendment" for the commodity operation record.  You will be given the opportunity to update both the commodity and entity information as part of this process. </span>'
    else:
        content = '<span id="tab-two-content">What permit are you applying for?</span>'
    bot_message = content
    chat_history.append((None, bot_message))
    return "", chat_history

def display_message(button_text):
    if button_text == 'Apply for and update the permit, registration, or notice.':
        return '**Press send to ask the ai more about this** Permits Online makes it easy for you to apply for and update the permit, registration, or notice you need to operate a TTB-regulated business. Once you have registered to use Permits Online, just follow the instructions and prompts for submitting your application package.  There is no fee to apply. Permits Online makes it easy for you to apply for and update the permit, registration, or notice you need to operate a TTB-regulated business.  Once you have registered to use Permits Online, just follow the instructions and prompts for submitting your application package.  There is no fee to apply.'
    elif button_text == 'Apply for a Certificate of Label Approval/Exemption (COLA).':
        return '**Press send to ask the ai more about this** COLAs Online makes it easy for you to apply for a Certificate of Label Approval/Exemption (COLA) for your product. Once you have registered, just follow the instructions and prompts for submitting your online application. In most cases, once your application is submitted electronically, you\'re done. There are no forms to sign and... there is no fee to apply. COLAs Online makes it easy for you to apply for a Certificate of Label Approval/Exemption (COLA) for your product. Once you have registered, just follow the instructions and prompts for submitting your online application. In most cases, once your application is submitted electronically, you\'re done. There are no forms to sign and... there is no fee to apply.'
    elif button_text == 'Your wine, distilled spirit, or beer/malt beverage may require formula approval or laboratory sample analysis':
        return '**Press send to ask the ai more about this** Your wine, distilled spirit, or beer/malt beverage may require formula approval or laboratory sample analysis before you may apply for a Certificate of Label Approval, or COLA. We require formula approval most commonly when a product has added flavoring or coloring materials. New to Formula Approval? A formula is a complete list of the ingredients used to make an alcohol beverage and a step-by-step description of how it\'s made. In some cases, we may also require laboratory analysis of the product. Your wine, distilled spirit, or beer/malt beverage may require formula approval or laboratory sample analysis before you may apply for a Certificate of Label Approval, or COLA. We require formula approval most commonly when a product has added flavoring or coloring materials. New to Formula Approval? A formula is a complete list of the ingredients used to make an alcohol beverage and a step-by-step description of how it\'s made. In some cases, we may also require laboratory analysis of the product. Your wine, distilled spirit, or beer/malt beverage may require formula approval or laboratory sample analysis before you may apply for a Certificate of Label Approval, or COLA. We require formula approval most commonly when a product has added flavoring or coloring materials. New to Formula Approval? A formula is a complete list of the ingredients used to make an alcohol beverage and a step-by-step description of how it\'s made. In some cases, we may also require laboratory analysis of the product. Your wine, distilled spirit, or beer/malt beverage may require formula approval or laboratory sample analysis before you may apply for a Certificate of Label Approval, or COLA. We require formula approval most commonly when a product has added flavoring or coloring materials. New to Formula Approval? A formula is a complete list of the ingredients used to make an alcohol beverage and a step-by-step description of how it\'s made. In some cases, we may also require laboratory analysis of the product.'

def predict(inp, history):
    #output = chain.run({'example_c':example,'input':inp})
    output = chain.run({'input':inp})
    history.append((inp, output))
    return "", history

def get_model_reply(message, chat_history):
   bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
   chat_history.append((message, bot_message))
   time.sleep(1)
   print(chat_history)
   return "", chat_history

def create_prompt_template():
    # Define the template for the conversation prompt
    template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

    Current conversation:
    {chat_history}
    Human: {input}
    AI:"""
  
    # Create the prompt from the template
    prompt = PromptTemplate(
        input_variables=["input", "chat_history"], template=template
    )
    return prompt

def initialize_chain() -> LLMChain:
    """Initialize and return the LangChain."""
    return load_chain()

def main():
    """Main function to run the application."""
    prompt = create_prompt_template()
    chain = initialize_chain()
    initialize_ui(chain, prompt)