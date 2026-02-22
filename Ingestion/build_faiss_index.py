from pathlib import Path
import numpy as np
import faiss


def build_index(embeddings_path: Path, index_output_path: Path):
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")

    emb = np.load(embeddings_path)

    # Ensure correct dtype for FAISS
    if emb.dtype != np.float32:
        emb = emb.astype(np.float32)

    if emb.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape {emb.shape}")

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product;
    index.add(emb)

    index_output_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_output_path))

    print(f"Loaded embeddings: {embeddings_path} shape={emb.shape} dtype={emb.dtype}")
    print(f"Built FAISS index: {index_output_path} ntotal={index.ntotal} dim={dim}")
