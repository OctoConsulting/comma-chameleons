
# Local Imports
from llm_utils import load_data_from_huggingface, create_vector_db, RAGUtils, Credentials
from app_utils import initialize_ui
#from settings import css
#from gradio import *


###### Build Vector DB ########

docs = load_data_from_huggingface('wikipedia',name="20220301.simple")

db = create_vector_db(docs)

###### Langchain ########

creds = Credentials('key.ini')
rag = RAGUtils(db,creds)
#local text
predict = rag()

###### Assistant Api ########

###### initialize a Gradio? ########
block = initialize_ui()
   
block.launch()#debug=True)

# if __name__ == "__main__":
#     main()
