import requests
import json

url = "http://localhost:7020/api/chat"
headers = {
    "Content-Type": "application/json"
}

data = {
    "model": "gemma2:9b",
    "messages": [{"role": "user", "content": "Tell me a very short joke."}],
    "stream": True
}

response = requests.post(url, headers=headers, json=data, stream=True)
for line in response.iter_lines():
    if line:
        chunk = json.loads(line.decode('utf-8'))
        if 'message' in chunk:
            print(chunk['message'].get('content', ''), end='', flush=True)
