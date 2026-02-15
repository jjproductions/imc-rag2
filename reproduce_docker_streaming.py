import requests
import json
import time
import sys

# Docker maps 8001 -> 8000
base_url = "http://localhost:8001"
# API key from .env
api_key = "local-key"

print(f"Checking health at {base_url}/health...")
try:
    resp = requests.get(f"{base_url}/health")
    if resp.status_code == 200:
        print("Health check OK")
    else:
        print(f"Health check failed: {resp.status_code} {resp.text}")
        sys.exit(1)
except requests.exceptions.ConnectionError:
    print(f"Could not connect to {base_url}. Is the docker container running and mapping port 8001?")
    sys.exit(1)

print(f"\nChecking auth at {base_url}/v1/models with key '{api_key}'...")
headers = {"Authorization": f"Bearer {api_key}"}
resp = requests.get(f"{base_url}/v1/models", headers=headers)
if resp.status_code == 200:
    print("Auth OK")
else:
    print(f"Auth failed: {resp.status_code}")
    print(resp.text)

url = f"{base_url}/v1/chat/completions"
data = {
    "model": "mistral-small3.2:24b-instruct-2506-q4_K_M",
    "messages": [{"role": "user", "content": "Tell me a short story."}],
    "stream": True
}

print(f"\nConnecting to {url} for streaming...")
start_time = time.time()
last_chunk_time = start_time

try:
    with requests.post(url, headers=headers, json=data, stream=True) as response:
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            sys.exit(1)

        print("Response received, reading stream...")
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                current_time = time.time()
                diff = current_time - last_chunk_time
                last_chunk_time = current_time
                
                if decoded_line.startswith("data: "):
                    content = decoded_line[6:]
                    if content == "[DONE]":
                        print("\n[DONE]")
                        break
                    try:
                        # Check for padding
                        if content.strip() == "": 
                            print(f"{diff:.4f}s: [PADDING]", end="", flush=True)
                            continue
                            
                        chunk = json.loads(content)
                        delta = chunk['choices'][0]['delta'].get('content', '') or chunk['choices'][0]['delta'].get('role', '')
                        print(f"{diff:.4f}s: {delta}", end="", flush=True)
                    except json.JSONDecodeError:
                        print(f"\nError decoding JSON: {content[:50]}...")
                else:
                    # Print weird lines (like : padding)
                    print(f"\nReceived raw line ({len(decoded_line)} bytes): {decoded_line[:50]}...")
except Exception as e:
    print(f"\nException: {e}")
