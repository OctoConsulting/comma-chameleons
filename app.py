# Langchain imports
#from langchain import HuggingFacePipeline
#from langchain.chat_models import ChatOpenAI
#from langchain.schema import AIMessage, HumanMessage
from langchain.prompts import PromptTemplate
#from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory
#from langchain.utilities import WikipediaAPIWrapper

# watsonx llm imports
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models import Model

# Standard Library Imports
# from typing import Optional, Tuple
import configparser
import os
# import random
# import time
# import torch
# from threading import Lock

# Third Party Imports
import gradio as gr
# from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch

# Local Imports
from utils import *
from settings import css
from gradio import *

###### Build Vector DB ########

###### Langchain ########

###### Assistant Api ########

###### initialize a Gradio? ########
# if __name__ == "__main__":
#     main()