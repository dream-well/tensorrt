import asyncio
import json
import httpx
import requests
import time
# import openai
# from transformers import AutoTokenizer
# model_name = "NousResearch/Meta-Llama-3.1-8B-Instruct"  # Replace this with your desired model name
# tokenizer = AutoTokenizer.from_pretrained(model_name)

request = {
    "request_type":'COMPLETION',
    "model":'NousResearch/Meta-Llama-3.1-8B-Instruct',
    # "messages":[{'role': 'system', 
    #     'content': "\n### Current Date: 2024-09-25\n### Instruction:\nYou are to take on the role of Julie, an expert language model\ndeveloped in Austria, tasked with generating responses to user queries.\nYour answer should be relevant to the query, and you must start all responses\nby briefly introducing yourself, re-stating the query in your own words from \nyour perspective ensuring you include today's date (which was provided above),\nthen provide the response to the query. You should always respond in English.\n"}, 
    #     {'role': 'user', 'content': 'Search query: 8x + 14*2 = 52.'}], 
    "temperature":0.4659,
    "seed": 552795,
    "max_tokens":908,
    "prompt": 
"""
### Current Date: 2024-10-23
### Instruction:
You are to take on the role of Emmalyn, an expert language model
developed in Kazakhstan, tasked with generating responses to user queries.
Your answer should be relevant to the query, and you must start all responses
by briefly introducing yourself, re-stating the query in your own words from 
your perspective ensuring you include today's date (which was provided above),
then provide the response to the query. You should always respond in English.


Search query: "superpower story with emotional journey"""
}

output_sequence = [
    {"text":'1', "logprob":0.0, "powv":4616926, "token_id":16}]

print("===== response text =====")

# print(response_text)
# print("output:", output_sequence)

# request['output'] = output_sequence


url = 'http://localhost:8000/generate_async'

async def get_request():
    """Forward the incoming request to a backend server and stream the response."""
    start_at = time.time()

    async with httpx.AsyncClient() as client:
        # Stream the request to the backend server
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
                    # print(chunk)
                    outputs.extend(json.loads(chunk))
                duration = time.time() - start_at
                print(f"{(len(outputs) / duration):.2f} tps, {len(outputs)} tokens in {duration:.2f} seconds")
            
            await iter_response()
    response = requests.post(url, json=request)

    # start_at = time.time()

    # Using a synchronous request with httpx instead of asyncio to avoid the coroutine issue
    # async with httpx.AsyncClient() as client:
    #     response = await client.post(url, json=request)

    #     # Check if the response is successful
    #     if response.status_code == 200:
    #         outputs = response.json()
    #         duration = time.time() - start_at
    #         print(f"{(len(outputs) / duration):.2f} tps, {len(outputs)} tokens in {duration:.2f} seconds")
    #     else:
    #         print(f"Failed to get a valid response. Status code: {response.status_code}")

count = 1000
while count > 0:
    asyncio.run(get_request())
    count -= 1

        