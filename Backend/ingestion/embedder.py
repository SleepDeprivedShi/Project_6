import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def load_texts_jsonl(path: Path) -> list[str]:
    texts = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            t = (obj.get("text") or "").strip()
            if t:
                texts.append(t)
    if not texts:
        raise ValueError(f"No non-empty texts found in {path}")
    return texts


def generate_embeddings(chunks_file: Path, output_file: Path, model_name: str):
    texts = load_texts_jsonl(chunks_file)
    print(f"Loaded {len(texts)} passages from {chunks_file}")

    model = SentenceTransformer(model_name)

    emb = model.encode(
        [f"passage: {t}" for t in texts],
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    )

    emb = emb.astype(np.float32)
    emb = l2_normalize(emb)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_file, emb)

    print(f"Saved embeddings to: {output_file}")
    print("Embeddings shape:", emb.shape, "dtype:", emb.dtype)

