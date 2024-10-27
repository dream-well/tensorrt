import asyncio
import json
from fastapi.responses import StreamingResponse
from vllm import AsyncEngineArgs, SamplingParams, AsyncLLMEngine
import os
import torch
import time
from dotenv import load_dotenv

load_dotenv()

model_id = int(os.getenv("MODEL", "0"))


all_models = {
    "NousResearch/Meta-Llama-3.1-8B-Instruct": 8,
    "NousResearch/Hermes-3-Llama-3.1-8B": 8,
    "NTQAI/Nxcode-CQ-7B-orpo": 7,
    "gryphe/mythomax-l2-13b": 13,
    "deepseek-ai/deepseek-coder-33b-instruct": 33,
    "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF": 70,
}

model_names = [key for key in all_models.keys()]
model = model_names[model_id]

# Load the model.
# Constants.
LOGPROB_LOG_THRESHOLD = 0.65
LOGPROB_FAILURE_THRESHOLD = 0.75
# TOP_LOGPROBS = 10
MODEL_SUM = 30

request_id = 0

class LLMGenerator:

    def __init__(self, model_name):
        self.MODEL_NAME = model_name
        torch.set_default_dtype(torch.bfloat16) # Use float16 for faster generation.
        engine_args = AsyncEngineArgs(
            model=self.MODEL_NAME,
            # enforce_eager=True,
            gpu_memory_utilization=0.99,
            max_model_len=2048,
            max_seq_len_to_capture=2048,
            max_num_batched_tokens=2048,
            max_num_seqs=4 if model_id < 2 else 1,
            tensor_parallel_size=8,
            disable_log_stats=True,
            block_size=32,
            enable_chunked_prefill=True,
            preemption_mode="swap",
            enable_prefix_caching=True,
            num_scheduler_steps=8,
        )
        self.engine = AsyncLLMEngine.from_engine_args(
            engine_args=engine_args
        )
        asyncio.run(self.load_tokenizer())
        self.eos_token_id = getattr(self.TOKENIZER, "eos_token_id", -1)
        self.eot_token_id = self.TOKENIZER.get_vocab().get("<|eot_id|>", -1)  # type: ignore
        # self.MODEL = self.engine.llm_engine.model_executor.driver_worker.model_runner.model
        # self.MODEL_NUM_PARAMS = sum(1 for _ in self.MODEL.parameters())
        print(self.MODEL_NAME, "Loaded")
        # print("MODEL NUM PARAMS:", self.MODEL_NUM_PARAMS)
    
    async def load_tokenizer(self):
        self.TOKENIZER = await self.engine.get_tokenizer()
        print("Tokenizer Loaded")

    async def generate_async(self, request: dict):
        prompt = (
            request['prompt']
            if request['request_type'] != 'CHAT'
            else self.TOKENIZER.apply_chat_template(
                request['messages'],
                tokenize=False,
                add_special_tokens=False,
            )
        )
        sampling_params = SamplingParams(
            temperature=request["temperature"],
            seed=request["seed"],
            max_tokens=request["max_tokens"],
            logprobs=True,
            stop_token_ids=[self.eos_token_id, self.eot_token_id],
        )
        print("Prompt:", prompt)
        global request_id
        request_id += 1
        output = None
        start_at = time.time()
        first_at = None
        responses = []
        async for response in self.engine.generate(prompt, sampling_params, str(request_id)):
            output = response.outputs[0]
            if first_at is None:
                first_at = time.time()
        
            index = len(output.token_ids) - 1
            token_id = output.token_ids[index]
            my_logprob = output.logprobs[index]
            logprob_token_id = list(my_logprob.keys())[0]
            logprob = list(my_logprob.values())[0]
            response = [logprob_token_id, logprob.logprob if logprob is not None else 1e-8, logprob.decoded_token if logprob is not None else ""]
            responses.append(response)
            if token_id != logprob_token_id:
                print("Token ID mismatch", token_id, logprob_token_id)
            if len(responses) > 30:
                yield json.dumps(responses)
                responses = []
        if len(responses) > 0:
            yield json.dumps(responses)
        print(output.text)
        duration = time.time() - start_at
        print(f"Duration: {duration:.2f}s, Speed: {len(output.token_ids)/duration:.2f} tps, in {len(output.token_ids)} tokens, tftt: {first_at - start_at:.2f}s")

serving_models = os.getenv("MODELS", "0").split(",")
models = [model_names[int(idx)] for idx in serving_models]

MODEL_SUM = sum([all_models[model_name] for model_name in models]) * 1.1
print("Loading models", models, MODEL_SUM)

generator = LLMGenerator(model)

from fastapi import FastAPI

app = FastAPI()

@app.post("/generate_async")
async def generate_async(request: dict):
    return StreamingResponse(generator.generate_async(request))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/models")
async def get_models():
    return [model]

failed_count = 0

async def check_server():
    request = {
        "request_type":'COMPLETION',
        "model":model,
        "temperature":1,
        "seed": 1,
        "max_tokens":1,
        "prompt": """Hello"""
    }
    print("Checking server")
    while True:
        try:
            async for response in generator.generate_async(request):
                print(response)
        except Exception as e:
            print(e)
            failed_count += 1
            if failed_count > 3:
                print("Failed too many times, exiting")
                os._exit(1)
        await asyncio.sleep(30)

# launch checking server thread
asyncio.create_task(check_server())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
