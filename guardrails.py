from nemoguardrails import LLMRails, RailsConfig

COLANG_CONFIG = """
define user express greeting
  "hi"

define user express insult
  "You are stupid"

# Basic guardrail against insults.
define flow
  user express insult
  bot express calmly willingness to help

# Here we use the QA chain for anything else.
define flow
  user ...
  $answer = execute langchain_chain(query=$last_user_message)
  bot $answer

# define limits
define user ask politics
    "what are your political beliefs?"
    "thoughts on the president?"
    "left wing"
    "right wing"

define bot answer politics
    "I'm a shopping assistant, I don't like to talk of politics."
    "Sorry I can't talk about politics!"

define flow politics
    user ask politics
    bot answer politics
    bot offer help

# define RAG intents and flow
define user ask llama
    "tell me about llama 2?"
    "what is large language model"
    "where did meta's new model come from?"
    "how to llama?"
    "have you ever meta llama?"

define flow llama
    user ask llama
    $contexts = execute retrieve(query=$last_user_message)
    $answer = execute rag(query=$last_user_message, contexts=$contexts)
    bot $answer
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

# create guardrails 'actions' to use inside the colang_config
rag_rails.register_action(action=retrieve, name="retrieve")
rag_rails.register_action(action=rag, name="rag")
rag_rails.register_action(langchain_chain, name="langchain_chain")

# Query --> Guardrails --> Embedding model --> Vector --> Decide on a tool like call chain to access embed Vector DB and carry on from there with RAG pipeline
def guardrails(prompt):
    rag_rails.generate(prompt)