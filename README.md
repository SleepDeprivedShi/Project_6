# RAG Pipeline

This project implements a modular Retrieval-Augmented Generation (RAG) pipeline in Python. It ingests documents, generates embeddings, builds a searchable index, and uses a large language model to answer questions based on the documents.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

*   Python 3.8+
*   A running Ollama instance with a model downloaded (e.g., `phi3:3.8b`)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt

4.  **Download the Embedding Model**: The project uses the `intfloat/e5-small` embedding model. This model needs to be downloaded manually and placed in the `Models/embeddings/e5-small/` directory.

    You can download the model by cloning its repository from Hugging Face:
    ```bash
    git clone https://huggingface.co/intfloat/e5-small Models/embeddings/e5-small
    ```

    Ensure that the `Models/embeddings/e5-small/` directory contains all the model files after cloning.
    ```

### Workflow

The project follows a standard Retrieval-Augmented Generation (RAG) pattern. The main script `run_pipeline.py` orchestrates the entire flow.

1.  **Data Ingestion**: The process starts with text data in `Storage/dummy_chunks.jsonl`.

2.  **Embedding Generation**: The `Ingestion/embedder.py` script reads the text chunks, and using a sentence-transformer model (from `Models/embeddings/e5-small`), it generates vector embeddings. These embeddings are saved as a NumPy array in `Storage/embeddings.npy`.

3.  **Index Creation**: The `Ingestion/build_faiss_index.py` script takes the embeddings from `Storage/embeddings.npy` and builds a FAISS index for efficient similarity searching. The index is saved to `Storage/index.faiss`.

4.  **Retrieval**: When a query is made (either in the `run_pipeline.py` test or interactively via `Retriever/retriever.py`), the query text is converted into a vector embedding. The FAISS index (`Storage/index.faiss`) is then searched to find the most similar text chunks from the original data (`Storage/dummy_chunks.jsonl`).

5.  **Augmented Generation**: The retrieved text chunks are used as context in a prompt that is sent to a large language model (`Models/llm/` with Ollama). The `Retriever/service.py` handles the request to the LLM, which then generates an answer based on the provided context.

### Running the Project

To run the entire pipeline in one go:
```bash
python run_pipeline.py
```
