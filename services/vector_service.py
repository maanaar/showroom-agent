"""
Vector similarity search over the vehicle catalog.

Requires the nomic-embed-text model pulled in Ollama:
    ollama pull nomic-embed-text

The index is built lazily on first call and cached in memory.
"""
import numpy as np
from typing import List

from langchain_ollama import OllamaEmbeddings

from services.data_service import _load, _vehicle_to_dict, COL_AVAILABLE

EMBED_MODEL = "nomic-embed-text"
EMBED_BASE_URL = "http://localhost:11434"

_embeddings: np.ndarray = None
_vehicles_cache: list = None
_embed_client: OllamaEmbeddings = None


def _get_client() -> OllamaEmbeddings:
    global _embed_client
    if _embed_client is None:
        _embed_client = OllamaEmbeddings(model=EMBED_MODEL, base_url=EMBED_BASE_URL)
    return _embed_client


def _vehicle_to_text(v: dict) -> str:
    fields = [
        v.get("name_ar"), v.get("name_en"), v.get("company"),
        v.get("type"), v.get("engine_cc"), v.get("engine_type"),
        v.get("transmission"), v.get("color"), v.get("condition"),
        v.get("notes"),
    ]
    return " ".join(str(f) for f in fields if f and str(f).lower() != "nan")


def _cosine_sim(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    return (matrix / norms) @ q


def _build_index() -> None:
    global _embeddings, _vehicles_cache
    df = _load()
    df = df[df[COL_AVAILABLE] == "متاح"]
    vehicles = [_vehicle_to_dict(row) for _, row in df.iterrows()]

    client = _get_client()
    texts = [_vehicle_to_text(v) for v in vehicles]
    vecs = client.embed_documents(texts)

    _embeddings = np.array(vecs, dtype=np.float32)
    _vehicles_cache = vehicles


def search_similar(query: str, k: int = 5) -> List[dict]:
    """Return the k most similar vehicles to the query string."""
    if _embeddings is None:
        _build_index()

    client = _get_client()
    q_vec = np.array(client.embed_query(query), dtype=np.float32)
    scores = _cosine_sim(q_vec, _embeddings)
    top_idx = np.argsort(scores)[::-1][:k]
    return [_vehicles_cache[i] for i in top_idx]


def invalidate_index() -> None:
    """Call this if the vehicle data changes so the index gets rebuilt."""
    global _embeddings, _vehicles_cache
    _embeddings = None
    _vehicles_cache = None
