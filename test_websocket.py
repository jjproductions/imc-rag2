import asyncio
import json
import sys

# Try to import websockets
try:
    import websockets
except ImportError:
    print("websockets library not found. Please install it: pip install websockets")
    sys.exit(1)

async def test_websocket():
    uri = "ws://localhost:8003/v1/chat/completions"
    
    # Payload
    payload = {
        # "model": "mistral",  # Omit to use default
        "messages": [
            {"role": "user", "content": "Hello! Please say 'WebSocket works' and nothing else."}
        ],
        "stream": True 
    }
    
    print(f"Connecting to {uri}...")
    headers = {"Authorization": "Bearer local-key"}
    try:
        async with websockets.connect(uri, additional_headers=headers) as websocket:
            print("Connected. Sending payload...")
            await websocket.send(json.dumps(payload))
            
            print("Waiting for response...")
            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    if "choices" in data and len(data["choices"]) > 0:
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        role = delta.get("role", "")
                        finish_reason = data["choices"][0].get("finish_reason")
                        
                        if role:
                            print(f"[Role: {role}]", end="", flush=True)
                        if content:
                            print(content, end="", flush=True)
                        if finish_reason:
                            print(f"\n[Finish reason: {finish_reason}]")
                            break
                    elif "error" in data:
                        print(f"\nError from server: {data['error']}")
                        break
                    else:
                         print(f"\nReceived: {message}")
                         
                except websockets.exceptions.ConnectionClosed as e:
                    print(f"\nConnection closed. Code: {e.code}, Reason: {e.reason}")
                    break
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())
