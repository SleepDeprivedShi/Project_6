#!/usr/bin/env python3
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Add current directory to path so we can import modules
sys.path.append(str(Path(__file__).parent))

from Backend.Ingestion import embedder, build_faiss_index
from Backend.Retriever import retriever, message_builder, service

# Configuration
CHUNKS_FILE = Path("Backend/Storage/dummy_chunks.jsonl")
EMBEDDINGS_FILE = Path("Backend/Storage/embeddings.npy")
INDEX_FILE = Path("Backend/Storage/index.faiss")
HISTORY_FILE = Path("Backend/Storage/chat_history.jsonl")

# Load model locally
MODEL_NAME = "Models/embeddings/e5-base"
LLM_NAME = "phi3:3.8b"

def run_test_pipeline():
    print("=== 1. Generating Embeddings ===")
    embedder.generate_embeddings(CHUNKS_FILE, EMBEDDINGS_FILE, MODEL_NAME)
    
    print("\n=== 2. Building FAISS Index ===")
    build_faiss_index.build_index(EMBEDDINGS_FILE, INDEX_FILE)
    
    print("\n=== 3. Testing Retrieval ===")
    # Load resources
    try:
        chunks, index = retriever.load_resources(CHUNKS_FILE, INDEX_FILE)
    except Exception as e:
        print(f"Error loading resources: {e}")
        return

    print("Loading model...")
    model = SentenceTransformer(MODEL_NAME)
    
    # Test query
    test_query = "What is the windows kernel?"
    print(f"\nQuery: {test_query}")
    results = retriever.retrieve(test_query, model, index, chunks, top_k=3)
    
    print("\nResults:")
    for res in results:
        print(f"[{res['score']:.4f}] {res['chunk_id']} ({res['page_info']}): {res['preview']}")

    # --- Generate Prompt for LLM ---
    print("\n=== 4. Generating Prompt (Ollama/Phi-3) ===")
    context_for_prompt = []
    for res in results:
        context_for_prompt.append({
            "text": res["full_text"],
            "chunk_id": res["chunk_id"],
            "page_start": res.get("page_start"), # optional
        })

    messages = message_builder.build_message(test_query, context_for_prompt)
    #TODO: format types

    response = service.request_ollama(LLM_NAME, messages)
    print(response)


if __name__ == "__main__":
    run_test_pipeline()

