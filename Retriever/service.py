import requests


def request_ollama(llm: str, messages: list[dict]):
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": llm,   # adjust to your actual model name
            "messages": messages,
            "stream": False
        }
    )

    response.raise_for_status()
    print(response.json()["message"]["content"])
    return response
