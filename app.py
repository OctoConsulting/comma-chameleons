
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
