# Local Imports
from llm_utils import load_vector_db, RAGUtils, Credentials
from app_utils import initialize_ui

# Initialize the RAG pipeline 
def initialize_rag(db, creds):
    return RAGUtils(db, creds)

# Launch the Gradio UI
def launch_gradio(predict):
    block = initialize_ui(predict)
    block.launch()

# Initialize the RAG pipeline and also the Launch the Gradio UI
def main():
    ###### Build Vector DB ########
    db = load_vector_db("./chroma_db_small")

    ###### Langchain ########
    creds = Credentials('key.ini')
    rag = initialize_rag(db,creds)
    predict = rag
    launch_gradio(predict)

if __name__ == "__main__":
    main()