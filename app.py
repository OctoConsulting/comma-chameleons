
# Local Imports
from llm_utils import load_data_from_huggingface, load_vector_db, RAGUtils, Credentials
from app_utils import initialize_ui
#from settings import css
#from gradio import *
import asyncio
import concurrent.futures


# ###### Build Vector DB ########

# docs = load_data_from_huggingface('wikipedia',name="20220301.simple")

# db = create_vector_db(docs)

# ###### Langchain ########

# creds = Credentials('key.ini')
# rag = RAGUtils(db,creds)
# #local text
# predict = rag()

# ###### Assistant Api ########

# ###### initialize a Gradio? ########
# block = initialize_ui()
   
# block.launch()#debug=True)

# # if __name__ == "__main__":
# #     main()



###### Assistant Api ########


# #local text
#predict = rag_task



def initialize_rag(db, creds):
    return RAGUtils(db, creds)

def launch_gradio(predict):
    block = initialize_ui(predict)
    block.launch()
    


#async def main():
def main():
    ###### Build Vector DB ########
    
    #docs = load_data_from_huggingface('wikipedia', name="20220301.simple")
    #db = create_vector_db(docs)
    db = load_vector_db("./chroma_db_wiki_base")
    ###### Langchain ########
    creds = Credentials('key.ini')
    rag = initialize_rag(db,creds)
    predict = rag
    launch_gradio(predict)
    
    #with concurrent.futures.ThreadPoolExecutor() as executor:
       # gradio_future = executor.submit(launch_gradio, predict)  # run Gradio in a separate thread
        
        # Initialize RAG and await any other async functions here
    #    rag_task = asyncio.create_task(initialize_rag(db, creds))
    #    predict = rag_task
        # Can add another async function here in the future for example
    #    gradio_future = executor.submit(launch_gradio, predict)  # run Gradio in a separate thread
    #    await rag_task

    # Optionally, handle the Gradio future result or exceptions here
    # e.g., gradio_result = gradio_future.result()

if __name__ == "__main__":
    #asyncio.run(main())
    main()