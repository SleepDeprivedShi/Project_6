
def build_message(query: str, context_chunks: list[dict], format_type="paragraph") -> list[dict]:

    SYSTEM_MESSAGE = """
You are a study assistant.

You must follow these rules strictly:

1. Use ONLY the provided CONTEXT to answer.
2. If the answer is not found in the context, reply exactly:
   Not found in material.
3. Always cite sources using:
   (chunk_id, page_number)
4. Follow the requested output format exactly.
5. Do not use outside knowledge.
"""

    messages = [ {"role": "system", "content": SYSTEM_MESSAGE}, ]
    #TODO: add history to the ocntext
    messages.append({
        "role" : "user",
        "content" : build_user_message(query, context_chunks, format_type)
    })

    return messages


def build_context_block(context_chunks: list[dict]) -> str:
    lines = [];
    for chunk in context_chunks:
        chunk_id = chunk["chunk_id"]
        page = chunk["page_start"]
        text = chunk["text"]

        lines.append(f"[{chunk_id} | page {page}]\n{text}")
    return "\n\n".join(lines)


def build_user_message(question: str, context_chunks: list[dict], format_type) -> str:
    context_block = build_context_block(context_chunks)

    return f""""
CONTEXT:
{context_block}

QUESTION:
{question}

FORMAT:
{format_type}
"""
