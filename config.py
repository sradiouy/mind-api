# config.py
import os

# --- Pinecone Configuration ---
# Load from environment variables (recommended) or replace None with your key
#PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
# Pinecone environment might be needed depending on specific client usage or future needs
#PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter") # Replace if needed
#INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "") # Set default 'mind' or load from env
#NAMESPACE = os.getenv("PINECONE_NAMESPACE", "") # Default to empty namespace

# --- OpenAI Configuration ---
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

EMBEDDING_MODEL = "text-embedding-3-large"

# --- API / Retrieval Configuration ---
TOP_K = 10 # Number of results to retrieve from Pinecone

# --- Flask Configuration (Optional) ---
# Example: For production, you'd set DEBUG=False
#DEBUG_MODE = os.getenv("FLASK_DEBUG", "True").lower() in ['true', '1', 't'] # Default to True for dev

# --- Input/Output Data Paths (Optional) ---
# Example if you move data files
# DATA_DIR = "data"
# INPUT_JSON_PATH = os.path.join(DATA_DIR, "pinecone_data_final_modules.json")