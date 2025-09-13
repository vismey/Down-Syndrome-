import requests

def query_ollama(model: str, prompt: str):
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,   # e.g., "mapler/gpt2"
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(url, json=payload)
    return response.json()[""]["content"]
