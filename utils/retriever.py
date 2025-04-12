# utils/retriever.py
from pinecone import Pinecone # <--- Quitamos ApiException de aquí
import config

# (Asumimos que pinecone_index se pasa desde app.py)

def query_pinecone(index: Pinecone.Index, query_vector: list[float], module_filter: str | None = None) -> list[dict]:
    """
    Queries the Pinecone index, optionally filtering by module,
    and returns formatted results.
    """
    if not query_vector:
        return []

    filter_dict = None
    if module_filter and isinstance(module_filter, str) and module_filter.strip():
        filter_dict = {"module": module_filter.strip()}
        print(f"DEBUG RETRIEVER: Applying Pinecone filter: {filter_dict}")
    else:
        print("DEBUG RETRIEVER: No module filter applied, searching all modules.")

    try:
        result = index.query(
            vector=query_vector,
            namespace=config.NAMESPACE,
            top_k=config.TOP_K,
            include_metadata=True,
            filter=filter_dict
        )

        matches = []
        for match in result.matches:
            metadata = match.metadata or {}
            matches.append({
                "id": match.id,
                "score": match.score,
                "text": metadata.get("text", ""),
                "source": metadata.get("source", "unknown"),
                "module": metadata.get("module", "unknown"),
            })
        return matches
    # except ApiException as e: <-- BLOQUE ELIMINADO
    #     print(f"Pinecone API Error during query: {e}")
    #     raise e
    except Exception as e: # <-- Este bloque atrapará ahora también los errores de Pinecone
        print(f"Unexpected error during query (may include Pinecone errors): {e}")
        raise e # Re-lanza para que lo atrape el manejador de Flask
