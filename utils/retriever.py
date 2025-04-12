# utils/retriever.py
from pinecone import Pinecone
from config import INDEX_NAME, NAMESPACE, TOP_K # Import from config

# Pinecone client can be initialized here or passed from app.py
# If initialized here, you'd need PINECONE_API_KEY from config too.
# Let's assume it's passed from app.py for this example to avoid multiple initializations.

def query_pinecone(index: Pinecone.Index, query_vector: list[float]) -> list[dict]:
    """
    Queries the Pinecone index and returns formatted results.
    """
    if not query_vector:
        return [] # Return empty list if no vector provided

    try:
        result = index.query(
            vector=query_vector,
            namespace=NAMESPACE, # Use namespace from config
            top_k=TOP_K,         # Use top_k from config
            include_metadata=True
        )

        # Format matches
        matches = []
        for match in result.matches:
            metadata = match.metadata or {} # Ensure metadata exists
            matches.append({
                "id": match.id, # Include the vector ID
                "score": match.score,
                "text": metadata.get("text", ""), # Safely get text
                "source": metadata.get("source", "unknown"), # Safely get source
                "module": metadata.get("module", "unknown"), # Example: include module
                # Add any other metadata fields you want to return
                # "topic": metadata.get("topic", "unknown"),
                # "tags": metadata.get("tags", []),
            })
        return matches
    except ApiException as e:
        print(f"Pinecone API Error during query: {e}")
        # Depending on desired API behavior, re-raise or return error indicator
        raise e # Re-raise to be caught by Flask error handler
    except Exception as e:
        print(f"Unexpected error during query: {e}")
        raise e