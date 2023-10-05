# Local Imports
from llm_utils import load_vector_db, RAGUtils, Credentials
from app_utils import initialize_ui
from nemo_guardrails import initialize_rails

# Initialize the RAG pipeline 
def initialize_rag(db, creds):
    return RAGUtils(db, creds)

# Launch the Gradio UI
def launch_gradio(predict):
    block = initialize_ui(predict)
    block.launch()

# Initialize the RAG Rails pipeline and also the Launch the Gradio UI
def main():
    launch_gradio(initialize_rails)

if __name__ == "__main__":
    main()