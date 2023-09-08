def web_scrap(url):
    from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
    loader = RecursiveUrlLoader(url=url)
    docs = loader.load
    return docs

def create_vector_db(docs):
    #Need to change to a WatsonX Encoder
    from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.vectorstores import Chroma
    
    # split it into chunks
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    docs_split = text_splitter.split_documents(docs)

    # create the open-source embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # load it into Chroma
    db = Chroma.from_documents(docs_split, embedding_function,persist_directory="./chroma_db")
    #db.persist()
    return db

def load_chain(db):
    
    from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
    from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
    from ibm_watson_machine_learning.foundation_models import Model
    #from langchain.memory import ConversationBufferMemory
    from langchain.prompts import PromptTemplate
    #from langchain.prompts.pipeline import PipelinePromptTemplate
    #from langchain.chains import LLMChain
    from langchain.chains import RetrievalQA
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
    
    prompt_template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    # Retrieve more documents with higher diversity- useful if your dataset has many similar documents
    
    retriever=db.as_retriever(search_type="mmr", search_kwargs={'k': 10, 'lambda_mult': 0.25})
    
    #TODO: Do we need this with WxA
    #memory = ConversationBufferMemory(memory_key='chat_history')
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,chain_type_kwargs=chain_type_kwargs)
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
        

def predict(inp, history):
    #output = chain.run({'example_c':example,'input':inp})
    output = chain.run({'input':inp})
    history.append((inp, output))
    return "", history

#For Testing
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

#def initialize_chain() -> LLMChain:
#    """Initialize and return the LangChain."""
#    return load_chain()

#def main():
#    """Main function to run the application."""
#    prompt = create_prompt_template()
#    chain = initialize_chain()
#    initialize_ui(chain, prompt)