import random
from vllm import LLM, SamplingParams
import os
from typing import List, Tuple
import math
import time

# Load the model.
MODEL_NAME = os.getenv("MODEL", "NousResearch/Meta-Llama-3.1-8B-Instruct")
# Constants.
LOGPROB_LOG_THRESHOLD = 0.65
LOGPROB_FAILURE_THRESHOLD = 0.85
TOP_LOGPROBS = 7

MODEL_WRAPPER = LLM(
    model=MODEL_NAME,
    enforce_eager=False,
    gpu_memory_utilization=0.8,
    max_model_len=4096,
    tensor_parallel_size=8
)

TOKENIZER = MODEL_WRAPPER.get_tokenizer()
MODEL = MODEL_WRAPPER.llm_engine.model_executor.driver_worker.model_runner.model
MODEL_NUM_PARAMS = sum(1 for _ in MODEL.parameters())

print("MODEL NUM PARAMS:", MODEL_NUM_PARAMS)

def verify_logprobs_random(request, input_text: str) -> Tuple[bool, str]:
    """
    Generate a handful of random outputs to ensure the logprobs weren't generated after the fact.
    """
    start_at = time.time()
    print("===== Verifying First Logprob =====")

    idx = 0

    # Generate a single token at each index, comparing logprobs.
    sampling_params = SamplingParams(
        temperature=request["temperature"],
        seed=request["seed"],
        max_tokens=1,
        logprobs=TOP_LOGPROBS,
    )
    full_text = input_text + "".join(
        [item["text"] for item in request['output'][0:idx]]
    )
    output = MODEL_WRAPPER.generate(
        [full_text], sampling_params, use_tqdm=False
    )[0].outputs[0]

    # The miner's output token should be in the logprobs...
    top_tokens = []
    for lp in output.logprobs:
        top_tokens += list(lp.keys())
    print("Time taken:", time.time() - start_at)
    if request['output'][idx]['token_id'] not in top_tokens:
        return False, top_tokens
    return (
        True,
        top_tokens
    )


def generate_logprobs_fast(
    request, input_text: str, input_tokens: List[int]
) -> Tuple[bool, str]:
    """
    Compare the produced logprob values against the ground truth, or at least
    the ground truth according to this particular GPU/software pairing.
    """

    # Set up sampling parameters for the "fast" check, which just compares input logprobs against output logprobs.
    sampling_params = SamplingParams(
        temperature=request["temperature"],
        seed=request["seed"],
        max_tokens=1,
        logprobs=1,
        prompt_logprobs=1,
    )

    # Generate output for a single token, which will return input logprobs based on prompt_logprobs=1
    full_text = input_text + "".join(
        [item["text"] for item in request['output']]
    )
    output = MODEL_WRAPPER.generate([full_text], sampling_params, use_tqdm=False)[0]
    assert output.prompt_logprobs is not None

    idxs = min(
        len(output.prompt_logprobs) - len(input_tokens) - 3,
        len(request['output']) - 1,
    )
    logprobs = []
    for idx in range(idxs):
        item = request['output'][idx]
        expected_logprob = output.prompt_logprobs[idx + len(input_tokens)].get(
            item['token_id']
        )
        if expected_logprob is not None:
            expected_logprob = expected_logprob.logprob
        else:
            expected_logprob = 0
        logprobs.append(expected_logprob)
    if request['request_type'] == 'CHAT':
        print("full text:", full_text)
        print("output sequence:", request["output"][0:5])
        print(logprobs[0:5])
    return logprobs

from fastapi import FastAPI

app = FastAPI()

@app.post("/powv")
async def powv(request: dict):
    # Tokenize the input sequence.
    input_text = (
        request['prompt']
        if request['request_type'] != 'CHAT'
        else TOKENIZER.apply_chat_template(
            request['messages'],
            tokenize=False,
            add_special_tokens=False,
        )
    )
    assert isinstance(input_text, str)
    if input_text.startswith(TOKENIZER.bos_token):
        input_text = input_text[len(TOKENIZER.bos_token) :]
    input_tokens = TOKENIZER(input_text).input_ids
    randprob_verified, top_tokens = verify_logprobs_random(request, str(input_text))
    if not randprob_verified:
        first_token = top_tokens[0]    
        print("Random logprobs failed. Using top token:", first_token)
        request['output'][0]['token_id'] = top_tokens[0]
        request['output'][0]['text'] = TOKENIZER.decode(top_tokens[0])
    # powv_list = generate_powv(request['output'], input_tokens)
    logprobs = generate_logprobs_fast(request, str(input_text), input_tokens)
    first_token = request['output'][0]['text']
    first_id = request['output'][0]['token_id']
    return {"logprobs": logprobs, "first_token": first_token, "first_id": first_id}

@app.post("/v1/chat/completions")
async def chat_completion(request: dict):
    print("===== Verifying First Logprob =====")

    idx = 0
    request['request_type'] = 'CHAT'
    input_text = (
        request['prompt']
        if request['request_type'] != 'CHAT'
        else TOKENIZER.apply_chat_template(
            request['messages'],
            tokenize=False,
            add_special_tokens=False,
        )
    )
    start_at = time.time()
    print(input_text)

    # Generate a single token at each index, comparing logprobs.
    sampling_params = SamplingParams(
        temperature=request["temperature"],
        seed=request["seed"],
        max_tokens=request["max_tokens"],
        logprobs=TOP_LOGPROBS,
    )
    full_text = input_text
    output = MODEL_WRAPPER.generate(
        [full_text], sampling_params, use_tqdm=False
    )[0].outputs[0]
    period = time.time() - start_at
    tps = len(output.token_ids) / period
    print(f"TPS: {tps}, Tokens: {len(output.token_ids)}, Time: {period}")
    return output

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
