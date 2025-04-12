# app.py
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone
import openai
# Import config variables and utility functions
import config
from utils.embedder import generate_embedding
from utils.retriever import query_pinecone

# === VALIDATE CONFIGURATION ===
if not config.PINECONE_API_KEY:
    raise ValueError("Pinecone API Key not found. Please set the PINECONE_API_KEY environment variable.")
if not config.OPENAI_API_KEY:
    raise ValueError("OpenAI API Key not found. Please set the OPENAI_API_KEY environment variable.")

# === FLASK APP & EXTENSIONS ===
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# === INITIALIZE EXTERNAL SERVICES ===
try:
    # Initialize Pinecone client (once)
    pc = Pinecone(api_key=config.PINECONE_API_KEY)

    # Check if index exists
    if config.INDEX_NAME not in [idx.name for idx in pc.list_indexes().indexes]:
         raise ValueError(f"Pinecone index '{config.INDEX_NAME}' not found.")

    # Get index object (once)
    pinecone_index = pc.Index(config.INDEX_NAME)
    print(f"Successfully connected to Pinecone index '{config.INDEX_NAME}'.")
    # Optionally print stats on startup
    # print(pinecone_index.describe_index_stats())

    # OpenAI client is initialized within embedder.py in this example

except Exception as e:
    # Log error appropriately
    print(f"ERROR: Failed to initialize external services: {e}")
    # Depending on severity, might want to exit or prevent app from starting fully
    pinecone_index = None # Ensure index is None if connection failed


# === API ENDPOINTS ===
@app.route('/health', methods=['GET'])
def health_check():
    """Basic health check endpoint."""
    # Add more checks if needed (e.g., Pinecone connection status)
    if pinecone_index is None:
         return jsonify({"status": "error", "message": "Pinecone connection failed"}), 503
    return jsonify({"status": "ok"})

@app.route("/rag/query", methods=["POST"])
def rag_query_endpoint():
    """
    Receives a query text, generates embedding, queries Pinecone,
    and returns relevant results.
    """
    if pinecone_index is None:
         return jsonify({"error": "Pinecone service unavailable"}), 503

    data = request.json
    if not data or "text" not in data or not data["text"].strip():
        return jsonify({"error": "Missing or empty 'text' field in JSON request"}), 400

    query_text = data["text"]
    print(f"Received query: {query_text[:100]}...") # Log received query (truncated)

    try:
        # 1. Generate embedding for the query text
        print("Generating embedding...")
        query_vector = generate_embedding(query_text)
        if not query_vector:
             # Handle case where embedding failed silently in the util function
             return jsonify({"error": "Failed to generate query embedding"}), 500
        print(f"Embedding generated (dim: {len(query_vector)})")

        # 2. Query Pinecone using the retriever utility
        print("Querying Pinecone...")
        results = query_pinecone(pinecone_index, query_vector)
        print(f"Retrieved {len(results)} results.")

        # 3. Return the results
        return jsonify({"results": results})

    except openai.APIError as e: # Be specific about potential errors
        print(f"OpenAI API Error: {e}")
        return jsonify({"error": f"OpenAI API Error: {e}"}), 500
    except Exception as e: # Catch-all for other unexpected errors
        # Log the full error for debugging
        print(f"Unexpected error processing query: {e}")
        # Return a generic error message to the client
        return jsonify({"error": "An internal server error occurred"}), 500


# === MAIN EXECUTION ===
if __name__ == "__main__":
    # Get host and port from environment variables or use defaults
    host = os.getenv("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_RUN_PORT", 5050))
    # Use debug mode from config
    app.run(host=host, port=port, debug=config.DEBUG_MODE)
