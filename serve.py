import argparse
import os
import time
from pathlib import Path
from typing import List

from cuda import cudart
from mpi4py.futures import MPIPoolExecutor
from transformers import AutoTokenizer

import tensorrt_llm
from tensorrt_llm import BuildConfig, Mapping, build, mpi_barrier
from tensorrt_llm.executor import GenerationExecutor, SamplingParams
from tensorrt_llm.models import LLaMAForCausalLM
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

# FastAPI app
app = FastAPI()

# Global variables to be used across functions
tokenizer = None
executor = None

class InputData(BaseModel):
    sampling_params: dict
    messages: List[dict]

def dataset():
    input_text = [
        'Born in north-east France, Soyer trained as a',
        "What is large language model?"
    ]
    return input_text

def build_and_run_llama(hf_model_dir, engine_dir, force_build, tp_size, rank):
    global tokenizer, executor  # Ensure the tokenizer and executor are accessible globally
    tensorrt_llm.logger.set_level('verbose')
    status, = cudart.cudaSetDevice(rank)
    assert status == cudart.cudaError_t.cudaSuccess, f"cuda set device to {rank} errored: {status}"

    ## Build engine
    build_config = BuildConfig(max_input_len=128,
                               max_seq_len=512,
                               opt_batch_size=8,
                               max_num_tokens=4096,
                               max_batch_size=16)
    # build_config.builder_opt = 0  # fast build for demo, pls avoid using this in production, since inference might be slower
    build_config.plugin_config.gemm_plugin = 'bfloat16'  # for fast build, tune inference perf based on your needs
    build_config.plugin_config.gpt_attention_plugin = 'bfloat16'  # for fast build, tune inference perf based on your needs
    build_config.plugin_config.context_fmha_type = 'enabled'  # for fast build, tune inference perf based on your needs
    mapping = Mapping(world_size=tp_size, rank=rank, tp_size=tp_size)
    if force_build:
        llama = LLaMAForCausalLM.from_hugging_face(hf_model_dir, mapping=mapping, dtype="bfloat16")
        engine = build(llama, build_config)
        engine.save(engine_dir)
    mpi_barrier()  # make sure every rank engine build finished
    tokenizer = AutoTokenizer.from_pretrained(hf_model_dir)
    ## Initialize tokenizer and executor
    myexecutor = GenerationExecutor.create(engine_dir)
    sampling_params = SamplingParams(max_new_tokens=200)
    print("Created executor")
    if rank == 0:
        for inp in dataset():
            stream_output = myexecutor.generate_async(
                tokenizer.encode(inp),
                sampling_params=sampling_params,
                streaming=True)
            for state in stream_output:
                print(
                    f"Output: {tokenizer.decode(state.outputs[0].token_ids)}"
                )
        import uvicorn
        global executor
        executor = myexecutor
        uvicorn.run(app, host="0.0.0.0", port=80)

    mpi_barrier()
    return True


async def generate_text_async(messages, max_tokens, seed, timeout=2.5):
    start_at = time.time()
    query = messages[1]['content'][14:]
    prompt = tokenizer.apply_chat_template(messages)
    sampling_params = SamplingParams(
        repetition_penalty=1.0,
        length_penalty=1.0,
        temperature=0.01,
        top_p=0.998,
        max_new_tokens=max_tokens - len(prompt),
        end_id=tokenizer.eos_token_id,
        stop_token_ids=[tokenizer.eos_token_id],
        random_seed=seed,
    )
    stream = executor.generate_async(prompt,
                               sampling_params=sampling_params, streaming=True)
    responses = []
    first_at = None
    yield ""
    output_str = ""
    for output in stream:
        if first_at is None:
            first_at = time.time()
        token = tokenizer.decode(output.outputs[0].token_ids[-1])
        responses.append(token)
        
        output_str += token
        if (time.time() - start_at > 0.5) and len(output_str) > 30:
            print("chunk", output_str)
            yield output_str
            output_str = ""
        
        if timeout < time.time() - start_at:
            print(f"timeout {time.time() - start_at}s")
            break

    if output_str != "":
        yield output_str
    output_str = "".join(responses)
    wps = len(output_str.split(" ")) / (time.time() - start_at)
    print(f"query: {query}")
    print("output:", output_str[:100])
    print(f"wps: {wps}, {len(output_str.split(' '))} words in {time.time() - start_at} seconds, first token: {first_at - start_at}")

async def generate_text(messages, max_tokens, seed, timeout=2.5):
    output_str = ""
    async for output in generate_text_async(messages, max_tokens, seed, timeout):
        output_str += output
    return output_str

@app.post("/generate")
async def generate(data: InputData):
    output_str = ""
    async for output in generate_text_async(data.messages, data.sampling_params['max_new_tokens'], data.sampling_params.get('seed', 0), data.sampling_params.get('timeout', 0.6)):
        output_str += output
    return output_str

@app.post("/generate_async")
def generate_async(data: InputData):
    stream = generate_text_async(data.messages, data.sampling_params['max_new_tokens'], data.sampling_params.get('seed', 0), data.sampling_params.get('timeout', 0.6))
    return StreamingResponse(stream, media_type="text/plain")


def parse_args():
    parser = argparse.ArgumentParser(description="Llama single model example")
    parser.add_argument(
        "--engine_dir",
        type=str,
        required=True,
        help="Directory to save and load the engine. When -c is specified, always rebuild and save to this dir. When -c is not specified, load engine when the engine_dir exists, rebuild otherwise"
    )
    parser.add_argument(
        "--hf_model_dir",
        type=str,
        required=True,
        help="Read the model data and tokenizer from this directory"
    )
    parser.add_argument(
        '-b', '--build',
        action='store_true',
        help='Force to rebuild the engine',
        default=False
    )
    parser.add_argument("-n",
                        "--tp_size",
                        type=int,
                        default=2,
                        help="TP size to run the model")
    return parser.parse_args()

def main(args):
    status, gpus = cudart.cudaGetDeviceCount()
    assert status == 0 and gpus >= args.tp_size, f"The test needs at least {args.tp_size} GPUs, skipping"

    if not os.path.exists(args.engine_dir):
        os.makedirs(args.engine_dir, exist_ok=True)

    ## Build engine in parallel
    with MPIPoolExecutor(max_workers=args.tp_size) as pool:
        results = pool.map(build_and_run_llama,
                           [args.hf_model_dir] * args.tp_size,
                           [args.engine_dir] * args.tp_size,
                           [args.build] * args.tp_size,
                           [args.tp_size] * args.tp_size, range(args.tp_size))
        for r in results:
            assert r

if __name__ == "__main__":
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_dir)
    main(args)
