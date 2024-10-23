import random
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
import requests
import threading
import time
import os 
import httpx

app = FastAPI()

all_models = [
    "NousResearch/Meta-Llama-3.1-8B-Instruct",
    "NousResearch/Hermes-3-Llama-3.1-8B",
    "NTQAI/Nxcode-CQ-7B-orpo",
    "gryphe/mythomax-l2-13b",
    "deepseek-ai/deepseek-coder-33b-instruct",
    # "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
]
# Target server information
servers = [
    {'url': 'http://localhost:9001', 'name': 's9001', 'healthy': True, 'models': [0]},
    {'url': 'http://localhost:9002', 'name': 's9002', 'healthy': True, 'models': [1]},
    {'url': 'http://localhost:9003', 'name': 's9003', 'healthy': True, 'models': [2]},
    {'url': 'http://localhost:9004', 'name': 's9004', 'healthy': True, 'models': [3]},
    # {'url': 'http://localhost:9005', 'name': 's9005', 'healthy': True, 'models': [4]}
]

HEALTH_CHECK_INTERVAL = 5  # seconds


@app.get("/health")
async def health_check():
    """Health check endpoint for the proxy server itself."""
    return {"status": "healthy"}

async def forward_request(request: Request, server_url: str):
    """Forward the incoming request to a backend server and stream the response."""
    # try:
    print(f"{server_url}{request.url.path}")
    async with httpx.AsyncClient() as client:
        # Stream the request to the backend server
        async with client.stream(
            request.method,
            url=f"{server_url}{request.url.path}",
            headers={key: value for key, value in request.headers.items()},
            content=await request.body() if request.method in ("POST", "PUT") else None,
            timeout=10
        ) as response_stream:
            
            async for chunk in response_stream.aiter_bytes():
                # print(chunk)
                yield chunk
    print("done")


def get_online_servers(model_name):
    online_servers = []
    model_idx = all_models.index(model_name)
    for server in servers:
        if model_idx in server['models'] and server['healthy']:
            online_servers.append(server['url'])
    if len(online_servers) > 0:
        return online_servers
    for server in servers:
        if 0 in server['models'] and server['healthy']:
            online_servers.append(server['url'])
    return online_servers

@app.post("/{path:path}")
async def proxy(request: Request, path: str):
    """Route the request to one of the healthy backend servers."""
    model = request.headers.get('model')
    online_servers = get_online_servers(model)
    one_server = random.choice(online_servers)
    print(f"Model: {model}")
    print(f"Redirect to {one_server}/{path}")
    return RedirectResponse(url=f'{one_server}/{path}')

def check_server_health():
    """Periodically check the health of both backend servers."""
    while True:
        for server in servers:
            try:
                response = requests.get(server['url'] + "/health", timeout=5)
                server['healthy'] = response.status_code == 200
            except requests.RequestException:
                if server['healthy'] == True:
                    print(f"Restart server {server['url']}")
                    # pm2 restart server['name']
                    os.system(f"pm2 restart {server['name']}")

                print(f"Server {server['url']} is unhealthy")
                server['healthy'] = False


        time.sleep(HEALTH_CHECK_INTERVAL)


def start_health_check_thread():
    """Start a background thread for health checking."""
    health_check_thread = threading.Thread(target=check_server_health, daemon=True)
    health_check_thread.start()


if __name__ == "__main__":
    # Start the health check thread
    start_health_check_thread()

    # Start the FastAPI app with Uvicorn server on port 8000
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
