# app.py
import os
import sys # Added for potentially flushing output
from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone
import openai # Added openai

# Import config variables and utility functions
import config # <-- Imports config.py and executes it
from utils.embedder import generate_embedding
from utils.retriever import query_pinecone

# === DEBUG: Configuration Values ===
# Let's print the values RIGHT AFTER config is imported to see what the app sees
print("--- STARTING CONFIGURATION DEBUG ---")
print(f"DEBUG CONFIG: PINECONE_API_KEY Loaded = {bool(config.PINECONE_API_KEY)}") # Just check if it exists
print(f"DEBUG CONFIG: OPENAI_API_KEY Loaded = {bool(config.OPENAI_API_KEY)}") # Just check if it exists
# Print the value that's causing the error, using repr() to show quotes and type
print(f"DEBUG CONFIG: INDEX_NAME = {repr(config.INDEX_NAME)}")
print(f"DEBUG CONFIG: PINECONE_ENVIRONMENT = {repr(config.PINECONE_ENVIRONMENT)}")
print(f"DEBUG CONFIG: NAMESPACE = {repr(config.NAMESPACE)}")
print(f"DEBUG CONFIG: DEBUG_MODE = {repr(config.DEBUG_MODE)}")
print("--- FINISHED CONFIGURATION DEBUG ---")
sys.stdout.flush() # Try to force output flushing

# === VALIDATE CONFIGURATION (Keep these checks) ===
if not config.PINECONE_API_KEY:
    raise ValueError("Pinecone API Key not found. Please set the PINECONE_API_KEY environment variable.")
if not config.OPENAI_API_KEY:
    raise ValueError("OpenAI API Key not found. Please set the OPENAI_API_KEY environment variable.")
# Add check for index name based on what we expect
if not config.INDEX_NAME or not isinstance(config.INDEX_NAME, str):
     raise ValueError(f"Pinecone Index Name is invalid or empty: {repr(config.INDEX_NAME)}. Check config.py and environment variables.")

# === FLASK APP & EXTENSIONS ===
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# === INITIALIZE EXTERNAL SERVICES ===
# This block will now run AFTER the debug prints
try:
    print("Attempting to initialize Pinecone client...") # Added log
    pc = Pinecone(api_key=config.PINECONE_API_KEY)

    print(f"Attempting to list indexes to check for '{config.INDEX_NAME}'...") # Added log
    index_list = pc.list_indexes()
    index_names = [index_info.name for index_info in index_list.indexes]
    print(f"Found indexes: {index_names}") # Added log

    if config.INDEX_NAME not in index_names:
         # Raise error using the value we logged earlier for clarity
         raise ValueError(f"Pinecone index '{config.INDEX_NAME}' (type: {type(config.INDEX_NAME).__name__}) not found in available indexes: {index_names}")

    print(f"Attempting to connect to index '{config.INDEX_NAME}'...") # Added log
    pinecone_index = pc.Index(config.INDEX_NAME)
    print(f"Successfully connected to Pinecone index '{config.INDEX_NAME}'.")

except Exception as e:
    print(f"ERROR: Failed to initialize external services: {e}")
    pinecone_index = None


# === API ENDPOINTS (Keep as before) ===
@app.route('/health', methods=['GET'])
def health_check():
    """Basic health check endpoint."""
    if pinecone_index is None:
         return jsonify({"status": "error", "message": "Pinecone connection failed"}), 503
    return jsonify({"status": "ok"})

@app.route("/rag/query", methods=["POST"])
def rag_query_endpoint():
    """Receives query, embeds, queries Pinecone, returns results."""
    if pinecone_index is None:
         return jsonify({"error": "Pinecone service unavailable"}), 503

    data = request.json
    if not data or "text" not in data or not data["text"].strip():
        return jsonify({"error": "Missing or empty 'text' field in JSON request"}), 400

    query_text = data["text"]
    print(f"Received query: {query_text[:100]}...") # Log received query (truncated)

    try:
        print("Generating embedding...")
        query_vector = generate_embedding(query_text)
        if not query_vector:
             return jsonify({"error": "Failed to generate query embedding"}), 500
        print(f"Embedding generated (dim: {len(query_vector)})")

        print("Querying Pinecone...")
        # Make sure ApiException is imported if you want this specific except block
        results = query_pinecone(pinecone_index, query_vector)
        print(f"Retrieved {len(results)} results.")

        return jsonify({"results": results})

    except openai.APIError as e:
        print(f"OpenAI API Error: {e}")
        return jsonify({"error": f"OpenAI API Error: {e}"}), 500
    except Exception as e:
        print(f"Unexpected error processing query: {e}")
        return jsonify({"error": "An internal server error occurred"}), 500


# === MAIN EXECUTION (Keep as before) ===
if __name__ == "__main__":
    host = os.getenv("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_RUN_PORT", 5050))
    app.run(host=host, port=port, debug=config.DEBUG_MODE)
