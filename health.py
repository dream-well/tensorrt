import asyncio
import json
import httpx
import requests
import time
import os

request = {
    "request_type":'COMPLETION',
    "model":'NousResearch/Meta-Llama-3.1-8B-Instruct',
    "temperature":1,
    "seed": 1,
    "max_tokens":1,
    "prompt": """Hello"""
}

output_sequence = [
    {"text":'1', "logprob":0.0, "powv":4616926, "token_id":16}]

print("===== response text =====")

url = 'http://localhost:8000/generate_async'

failed_count = 0

async def get_request():
    global failed_count
    """Forward the incoming request to a backend server and stream the response."""
    start_at = time.time()

    async with httpx.AsyncClient() as client:
        # Stream the request to the backend server
        try:
            async with client.stream(
                'POST',
                url=url,
                content=json.dumps(request),
                headers={'model': request['model']},
                follow_redirects=True,
                timeout=10
            ) as response_stream:
                
                # Stream the response from the backend
                async def iter_response():
                    outputs = []
                    async for chunk in response_stream.aiter_bytes():
                        print(chunk)
                        outputs.extend(json.loads(chunk))
                    duration = time.time() - start_at
                    print(f"{(len(outputs) / duration):.2f} tps, {len(outputs)} tokens in {duration:.2f} seconds")
                
                await iter_response()
            failed_count = 0
        except Exception as e:
            print(e)
            failed_count += 1
            if failed_count > 3:
                print("Failed too many times, exiting")
                os.system("pm2 restart start_vllm")
                failed_count = 0
            time.sleep(120)
    requests.post(url, json=request)


# count = 1000
while True:
    asyncio.run(get_request())
    time.sleep(20)

        
