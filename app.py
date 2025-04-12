# app.py
import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone
import openai

# Import config variables and utility functions
import config
from utils.embedder import generate_embedding
from utils.retriever import query_pinecone

# === DEBUG Configuration Values ===
# ... (tus prints de depuración) ...

# === VALIDATE CONFIGURATION ===
# ... (tus validaciones) ...

# === FLASK APP & EXTENSIONS ===
app = Flask(__name__)
CORS(app)

# === INITIALIZE EXTERNAL SERVICES ===
# (Mantenemos el try-except general aquí)
try:
    print("Attempting to initialize Pinecone client...")
    pc = Pinecone(api_key=config.PINECONE_API_KEY)

    print(f"Attempting to list indexes to check for '{config.INDEX_NAME}'...")
    index_list = pc.list_indexes()
    index_names = [index_info.name for index_info in index_list.indexes]
    print(f"Found indexes: {index_names}")

    if config.INDEX_NAME not in index_names:
         raise ValueError(f"Pinecone index '{config.INDEX_NAME}'...") # Mensaje de error

    print(f"Attempting to connect to index '{config.INDEX_NAME}'...")
    pinecone_index = pc.Index(config.INDEX_NAME)
    print(f"Successfully connected to Pinecone index '{config.INDEX_NAME}'.")

except Exception as e: # Este except atrapará errores de inicialización
    print(f"ERROR: Failed to initialize external services: {e}")
    pinecone_index = None


# === API ENDPOINTS ===
@app.route('/health', methods=['GET'])
def health_check():
    if pinecone_index is None:
         return jsonify({"status": "error", "message": "Pinecone connection failed"}), 503
    return jsonify({"status": "ok"})

@app.route("/rag/query", methods=["POST"])
def rag_query_endpoint():
    if pinecone_index is None:
         return jsonify({"error": "Pinecone service unavailable"}), 503

    data = request.json
    if not data or "text" not in data or not data["text"].strip():
        return jsonify({"error": "Missing or empty 'text' field in JSON request"}), 400

    query_text = data["text"]
    module_to_filter = data.get("module")
    if module_to_filter and isinstance(module_to_filter, str):
        module_to_filter = module_to_filter.strip().lower()
        if not module_to_filter: module_to_filter = None
    else:
         module_to_filter = None

    print(f"Received query: '{query_text[:100]}...' | Module Filter: {module_to_filter}")

    try:
        print("Generating embedding...")
        query_vector = generate_embedding(query_text)
        if not query_vector:
             return jsonify({"error": "Failed to generate query embedding"}), 500
        print(f"Embedding generated (dim: {len(query_vector)})")

        print("Querying Pinecone...")
        # Llamamos a la función actualizada que ya no tiene except ApiException
        results = query_pinecone(
            index=pinecone_index,
            query_vector=query_vector,
            module_filter=module_to_filter
        )
        print(f"Retrieved {len(results)} results.")

        return jsonify({"results": results})

    except openai.APIError as e: # Mantenemos el de OpenAI
        print(f"OpenAI API Error: {e}")
        return jsonify({"error": f"OpenAI API Error: {e}"}), 500
    # except ApiException as e: <-- BLOQUE ELIMINADO
    #    print(f"Pinecone API Error: {e}")
    #    return jsonify({"error": f"Pinecone API Error: {e}"}), 500
    except Exception as e: # <-- Este atrapará todo lo demás, incluyendo errores de Pinecone
        print(f"Unexpected error processing query: {e}")
        return jsonify({"error": "An internal server error occurred"}), 500


# === MAIN EXECUTION ===
if __name__ == "__main__":
    host = os.getenv("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_RUN_PORT", 5050))
    app.run(host=host, port=port, debug=config.DEBUG_MODE)
