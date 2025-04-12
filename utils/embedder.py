# utils/embedder.py
import openai
from config import OPENAI_API_KEY, EMBEDDING_MODEL # Import from config

# Initialize OpenAI client (can be done here or globally in app.py)
# Ensure the key is loaded before initializing
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API Key not found. Please set the OPENAI_API_KEY environment variable.")
openai.api_key = OPENAI_API_KEY

def generate_embedding(text: str) -> list[float]:
    """Generates an embedding vector for the given text using OpenAI."""
    if not text:
        # Or raise an error, depending on desired behavior
        return []

    try:
        response = openai.embeddings.create(
            input=[text],
            model=EMBEDDING_MODEL
        )
        return response.data[0].embedding
    except Exception as e:
        # Consider more specific error handling and logging
        print(f"Error generating embedding: {e}")
        # Re-raise or return None/empty list based on desired handling
        raise e