from langchain.vectorstores import Chroma 
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods
from langchain.chains import (
        StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
    )
from langchain.memory import ChatMessageHistory
from langchain.schema import Document
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models import Model
from typing import List, Iterable
from pydantic import BaseModel, Field
import json
import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
logging.getLogger("langchain.retrievers.document_compressors.embeddings_filter").setLevel(logging.INFO)

#def web_scrap(url):
#    loader = RecursiveUrlLoader(url=url, extractor=lambda x: Soup(x, "html.parser").text)
#    docs = loader.load
#    return docs

#wierd dependecy on dill
def load_data_from_huggingface(path,name=None,page_content_column='text', max_len=20):
    #LangChain Wrapper does not support splits and assumes text context https://github.com/langchain-ai/langchain/issues/10674
    from langchain.document_loaders import HuggingFaceDatasetLoader
    loader = HuggingFaceDatasetLoader(path, page_content_column, name)
    docs = loader.load()
    if len(docs) < max_len:
        return docs[:max_len]
    else:
        return docs


def create_vector_db(docs):
    #Assumes input in LangChain Doc Format
    #[Document(page_content='...', metadata={'source': '...'})]
    #split docs into chunks
    from langchain.text_splitter import TokenTextSplitter #requires tiktoken
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
    text_splitter = TokenTextSplitter.from_huggingface_tokenizer(tokenizer,chunk_size=200, chunk_overlap=20)
    docs_split = text_splitter.transform_documents(docs)
    
    ###Get Encodder Model
    from langchain.embeddings import HuggingFaceBgeEmbeddings #requires sentence-transformers
    
    #This was not working in Wx Notebook
    #try:
    #    import torch
    #    is_cuda_available = torch.cuda.is_available()
    #except ImportError:
    #    is_cuda_available = False

    #device = 'cuda' if is_cuda_available else 'cpu'
    #normalize_embeddings = is_cuda_available
    device ='cuda'
    normalize_embeddings = True
    
    
    model_name = "BAAI/bge-large-en-v1.5"
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': normalize_embeddings}
    
    bge_hf = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    embedding_function= bge_hf
    
    # embed and load it into Chroma
    db = Chroma.from_documents(docs_split, embedding_function)#,persist_directory="./chroma_db_wiki")
    #db.persist()
    return db

# Output parser will split the LLM result into a list of queries
#Pydantic requires classes
class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
    lines: List[str] = Field(description="Lines of text")


class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)


class Credentials:
    def __init__(self,cfg_file=None, key=None, project=None, url=None):
        if cfg_file:
                self._load_credentials_from_file(cfg_file)
        # If key and project and url are provided, set attributes directly.
        elif key and project and url:
            self.watsonx_project_id = project
            self.watsonx_key = key
            self.watsonx_url = url
            
    def _load_credentials_from_file(self, cfg_file):
        config = configparser.ConfigParser()
        
        # Check if file exists.
        file_path = os.path.join(os.path.dirname(__file__), cfg_file)
        if not os.path.exists(file_path):
            raise SystemExit(f"Configuration file '{cfg_file}' does not exist.")
        
        config.read(file_path)
        
        try:
            self.watsonx_project_id = config['WATSONX_CREDS']['project_id']
            self.watsonx_key = config['WATSONX_CREDS']['key']
            self.watsonx_url = config['WATSONX_CREDS']['url']
        except KeyError:
            raise SystemExit("Could not find key in the configuration file.")



class RAGUtils:
    
    def __init__(self, db, credentials: Credentials = None):
        self._db = db
        self._creds = credentials
        self._history = ChatMessageHistory()
        #Load Chain
        try:
            #Low Temp Model Low Output (Faster/Cheaper) Model for RAG 
            self._llm_llama_temp_0p1 = self._get_llama(temp=0.2,max_new_tokens=300)
            #get a mid temp high output model for the QA
            self._llm_llama_temp_0p5 = self._get_llama(temp=0.5,max_new_tokens=1000)
            #The RAG
            self._chain = self._load_chain()
        except Exception as e:
            raise SystemExit(f"Error loading chain: {e}")

    def __call__(self,inp, chat_history):
        #Gradio Need to test WxA       
        for human, ai in chat_history:
            self._history.add_user_message(human)
            self._history.add_ai_message(ai)
        print(self._history.messages)
        output = self._chain.run({'question':inp, 'chat_history':self._history.messages})
        self._history.add_user_message(inp)
        self._history.add_ai_message(output)
        return "", self._history.messages

 
    
    def _get_llama(self,temp, max_new_tokens):
        # Found that GREEDY was better than SAMPLE to stay on task and get stop tokens
        # Temp was best tuning parameter but higher than .5 it was hallucinate 
        # Bigger output for RAG, smaller for final to be more direct for WxA
        params = {
            GenParams.MAX_NEW_TOKENS: max_new_tokens,
            GenParams.MIN_NEW_TOKENS: 1,
            GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
            GenParams.TEMPERATURE: temp,
            GenParams.TOP_K: 50,
            GenParams.TOP_P: .95
        }
      
        llama = ModelTypes.LLAMA_2_70B_CHAT #oh yeah

        llama_model= Model(
        model_id=llama,
        params=params,
        credentials={'url':self._creds.watsonx_url,'apikey':self._creds.watsonx_key},
        project_id=self._creds.watsonx_project_id)
        
        #LangChain Ready
        return WatsonxLLM(model=llama_model)
    
    def _get_retriever_chain(self):      
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate three
            different versions of the given user question to retrieve relevant documents from a vector 
            database. By generating multiple perspectives on the user question, your goal is to help
            the user overcome some of the limitations of the distance-based similarity search. 
            Provide these alternative questions seperated by newlines.
            Original question: {question}""",
        )

        base_retriever = self._db.as_retriever()
        embeddings = self._db.embeddings
        embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.75)
        compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter, base_retriever=base_retriever)

        #too slow
        #compressor = LLMChainExtractor.from_llm(llm_llama_temp_0p1)
        #compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)

        output_parser = LineListOutputParser()
        
        retriever_llm_chain = LLMChain(llm=self._llm_llama_temp_0p1, prompt=QUERY_PROMPT, output_parser=output_parser)
        multi_retriever = MultiQueryRetriever(
            retriever=compression_retriever, llm_chain=retriever_llm_chain, parser_key="lines"
        )  # "lines" is the key (attribute name) of the parsed output
        return multi_retriever
    
    def _get_stuff_chain(self):
            # This controls how each document will be formatted. Specifically,
        # it will be passed to the stuff chain - see that function for more details.
        document_prompt = PromptTemplate(
            input_variables=["page_content"],
            template="{page_content}"
        )
        document_variable_name = "context"
        sum_prompt = PromptTemplate.from_template(
            "Summarize this content: {context}"
        )
        sum_llm_chain = LLMChain(llm=self._llm_llama_temp_0p1, prompt=sum_prompt)
        stuff_chain = StuffDocumentsChain(
            llm_chain=sum_llm_chain,
            document_prompt=document_prompt,
            document_variable_name=document_variable_name
        )

        return stuff_chain

    def _get_qa_chain(self):
        # This controls how the standalone question is generated.
        # Should take `chat_history` and `question` as input variables.
        template = (
            "Combine the chat history and follow up question into "
            "a standalone question. Chat History: {chat_history}"
            "Follow up question: {question}"
        )
        prompt = PromptTemplate.from_template(template)
        
        question_generator_chain = LLMChain(llm=self._llm_llama_temp_0p5, prompt=prompt)
        return question_generator_chain


    def _load_chain(self):
    
        chain = ConversationalRetrievalChain(
            combine_docs_chain=self._get_stuff_chain(),
            retriever=self._get_retriever_chain(),
            question_generator=self._get_qa_chain(),
    )
        return chain


#For Testing
def get_model_reply(message, chat_history):
   bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
   chat_history.append((message, bot_message))
   time.sleep(1)
   print(chat_history)
   return "", chat_history

