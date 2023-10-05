from nemoguardrails import LLMRails, RailsConfig
from llm_utils import load_vector_db, RAGUtils, Credentials
import os

os.environ["OPENAI_API_KEY"] = ""

COLANG_CONFIG = """
define user express greeting
  "hi"
define user express insult
  "You are stupid"
# Basic guardrail against insults.
define flow
  user express insult
  bot express calmly willingness to help
define flow 
  user express greeting
  bot says exactly "hello human"
# define limits
define user ask politics
    "what are your political beliefs?"
    "thoughts on the president?"
    "left wing"
    "right wing"
define bot answer politics
    "I'm an llm assistant, I don't like to talk of politics."
    "Sorry I can't talk about politics!"
define flow politics
    user ask politics
    bot answer politics
    bot offer help
# Here we use the QA chain for anything else.
define flow
  user ...
  $answer = execute initialize_rag(inp=$last_user_message, history=$contexts)
  bot $answer
# define RAG intents and flow
#define user ask llama
#    "tell me about llama 2?"
#    "what is large language model"
#    "where did meta's new model come from?"
#    "how to llama?"
#    "have you ever meta llama?"
#define flow llama
#    user ask llama
#    $contexts = execute retrieve(query=$last_user_message)
#    $answer = execute rag(query=$last_user_message, contexts=$contexts)
#    bot $answer
"""

YAML_CONFIG = """
models:
  - type: main
    engine: openai
    model: text-davinci-003
"""

# initialize rails config
config = RailsConfig.from_content(COLANG_CONFIG, YAML_CONFIG)

# create rails
rag_rails = LLMRails(config)

# Initialize the RAG pipeline 
def initialize_rag(inp, history):
    ###### Build Vector DB ########
    db = load_vector_db("./chroma_db_wiki_base")
    
    ###### Langchain ########
    creds = Credentials('key.ini')

    rag_utils = RAGUtils(db=db, credentials=creds)
    return rag_utils(inp=inp, history=history)

# create guardrails 'actions' to use inside the colang_config
# rag_rails.register_action(action=retrieve, name="retrieve")
# rag_rails.register_action(action=rag, name="rag")
rag_rails.register_action(action=initialize_rag, name="initialize_rag")

# Query --> Guardrails --> Embedding model --> Vector --> Decide on a tool like call chain to access embed Vector DB and carry on from there with RAG pipeline
def initialize_rails(input_message, current_chat):
    # The user's message is already in the chat, so only need to add the bot's response
    bot_response = rag_rails.generate(prompt=input_message)
    bot_message = (input_message, "Bot: " + bot_response)
    current_chat.append(bot_message)
    # input_message = ""  # Clear the input_message for the next user input
    return input_message, current_chat

