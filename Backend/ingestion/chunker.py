import json
from pathlib import Path


CHUNK_SIZE = 10000
CHUNK_OVERLAP = 1500
STEP = CHUNK_SIZE - CHUNK_OVERLAP


def chunk_text(text: str, doc_id: str = "doc") -> list[dict]:
    chunks = []
    start = 0
    end = CHUNK_SIZE
    index = 0

    while start < len(text):
        chunk_text_slice = text[start:end]
        chunk_id = f"{doc_id}_{index}"
        chunks.append({
            "text": chunk_text_slice,
            "chunk_id": chunk_id,
            "char_start": start,
            "char_end": end,
        })
        start += STEP
        end = start + CHUNK_SIZE
        index += 1

    return chunks


def save_chunks(chunks: list[dict], output_file: Path):
    with output_file.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")