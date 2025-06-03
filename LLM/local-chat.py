import requests
import json

def ask_ollama_stream(prompt, model='llama3.1:8b'):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
    }

    with requests.post(
        "http://localhost:11434/api/generate", json=payload, stream=True
    ) as response:
        if response.status_code != 200:
            return f"Error: Server responded with status {response.status_code}"
        
        full_response = ""
        for line in response.iter_lines():
            if line:
                content = line.decode("utf-8").removeprefix("data: ")
                try:
                    chunk = json.loads(content)
                    partial = chunk.get("response", "")
                    print(partial, end="", flush=True)
                    full_response += partial
                except Exception as e:
                    print(f"\n[Error parsing line]: {content}")
        print()
        return full_response
