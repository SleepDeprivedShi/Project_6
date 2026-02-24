import json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def load_chunks_jsonl(path: Path):
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = (obj.get("text") or "").strip()
            if not text:
                continue
            chunks.append(obj)
    if not chunks:
        raise ValueError(f"No usable chunks found in {path}")
    return chunks


def get_page_info(obj: dict) -> str:
    # supports either page or page_start/page_end
    page = obj.get("page")
    ps = obj.get("page_start", page)
    pe = obj.get("page_end", page)
    if ps is None:
        return ""
    if pe is not None and pe != ps:
        return f"pages {ps}-{pe}"
    return f"page {ps}"


def load_resources(chunks_path: Path, index_path: Path):
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks file: {chunks_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index file: {index_path}")

    chunks = load_chunks_jsonl(chunks_path)
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    index = faiss.read_index(str(index_path))
    print(f"Loaded FAISS index from {index_path} (ntotal={index.ntotal})")

    # Safety check: index rows must match chunk count
    if index.ntotal != len(chunks):
        print("\nWARNING: index.ntotal != number of chunks!")
        print("This usually means chunks.jsonl order/count changed after building the index.")
        print(f"index.ntotal={index.ntotal}, chunks={len(chunks)}")
        print("Rebuild embeddings + index to fix.\n")

    return chunks, index



def retrieve(query: str, model: SentenceTransformer, index: faiss.Index, chunks: list, top_k: int = 5) -> List[Dict[str, Any]]:
    # Prefix 'query: ' for e5 models
    q_vec = model.encode([f"query: {query}"], convert_to_numpy=True, show_progress_bar=False)
    q_vec = l2_normalize(q_vec.astype(np.float32))

    scores, ids = index.search(q_vec, top_k)

    results = []
    for rank, (score, idx) in enumerate(zip(scores[0].tolist(), ids[0].tolist()), start=1):
        if idx == -1:
            continue
        if idx < 0 or idx >= len(chunks):
            continue

        obj = chunks[idx]
        chunk_id = obj.get("chunk_id", f"line{idx+1}")
        page_info = get_page_info(obj)
        preview = obj["text"][:240].replace("\n", " ").strip()
        if len(obj["text"]) > 240:
            preview += "…"

        results.append({
            "rank": rank,
            "score": score,
            "chunk_id": chunk_id,
            "page_info": page_info,
            "preview": preview,
            "full_text": obj["text"],
            "page_start": obj.get("page_start"),
            "page_end": obj.get("page_end")
        })
    return results
