import argparse
import os
import time
from pathlib import Path
from typing import List, Optional

from cuda import cudart
from mpi4py.futures import MPIPoolExecutor
import tensorrt_llm.bindings
from transformers import AutoTokenizer

import tensorrt_llm
from tensorrt_llm import BuildConfig, Mapping, build, mpi_barrier
from tensorrt_llm.executor import GenerationExecutor, SamplingParams
from tensorrt_llm.models import LLaMAForCausalLM
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse, StreamingResponse
import traceback
import json

all_models = {
    "NousResearch/Meta-Llama-3.1-8B-Instruct": 8,
    "NousResearch/Hermes-3-Llama-3.1-8B": 8,
    # "NTQAI/Nxcode-CQ-7B-orpo": 7,
    # "gryphe/mythomax-l2-13b": 13,
    # "deepseek-ai/deepseek-coder-33b-instruct": 33,
    # "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF": 70,
}

TOTAL_SIZE = sum(all_models.values())
BUFFER_SIZE = 30

# FastAPI app
app = FastAPI()

# Global variables to be used across functions
tokenizer = None
executor = None
tps_list = []
is_exit = False

class InputData(BaseModel):
    request_type: str
    prompt: Optional[str] = None
    messages: Optional[List[dict]] = None
    seed: int
    max_tokens: int
    temperature: float

def dataset():
    input_text = [
        'Born in north-east France, Soyer trained as a',
        "What is large language model?"
    ]
    return input_text

def build_and_run_llama(model, hf_model_dir, engine_dir, force_build, tp_size, port, rank):
    global tokenizer, executor  # Ensure the tokenizer and executor are accessible globally
    tensorrt_llm.logger.set_level('verbose')
    status, = cudart.cudaSetDevice(rank)
    assert status == cudart.cudaError_t.cudaSuccess, f"cuda set device to {rank} errored: {status}"

    ## Build engine
    build_config = BuildConfig(max_input_len=256,
                               max_seq_len=2400,
                               opt_batch_size=8,
                               max_num_tokens=2200,
                               max_batch_size=8,
                               max_prompt_embedding_table_size=24,
                               builder_opt=10,

                            #    gather_generation_logits=True,
                               )
    # build_config.builder_opt = 0  # fast build for demo, pls avoid using this in production, since inference might be slower
    build_config.plugin_config.gemm_plugin = 'bfloat16'  # for fast build, tune inference perf based on your needs
    build_config.plugin_config.gpt_attention_plugin = 'bfloat16'  # for fast build, tune inference perf based on your needs
    build_config.plugin_config.nccl_plugin = 'bfloat16'  # for fast build, tune inference perf based on your needs
    # build_config.kv_cache_type = tensorrt_llm.builder.KVCacheType.CONTINUOUS  # for fast build, tune inference perf based on your needs
    build_config.plugin_config.context_fmha = True  # for fast build, tune inference perf based on your needs
    build_config.plugin_config.paged_kv_cache = True  # for fast build, tune inference perf based on your needs
    build_config.plugin_config.tokens_per_block = 512
    build_config.plugin_config._remove_input_padding = True
    # build_config.plugin_config._use_paged_context_fmha = True  # for fast build, tune inference perf based on your needs
    build_config.kv_cache_type = tensorrt_llm.builder.KVCacheType.PAGED  # for fast build, tune inference perf based on your needs
    build_config.plugin_config._use_paged_context_fmha = True  # for fast build, tune inference perf based on your needs
    mapping = Mapping(world_size=tp_size, rank=rank, tp_size=tp_size)
    if force_build:
        llama = LLaMAForCausalLM.from_hugging_face(hf_model_dir, mapping=mapping, dtype="bfloat16")
        engine = build(llama, build_config)
        engine.save(engine_dir)
        return True
    mpi_barrier()  # make sure every rank engine build finished
    tokenizer = AutoTokenizer.from_pretrained(hf_model_dir)
    ## Initialize tokenizer and executor
    config = tensorrt_llm.bindings.executor.ExecutorConfig(
        batching_type=tensorrt_llm.bindings.executor.BatchingType.INFLIGHT,
        kv_cache_config=tensorrt_llm.bindings.executor.KvCacheConfig(
            enable_block_reuse=True,
            free_gpu_memory_fraction=0.9,
            max_tokens=32768
        ),
        normalize_log_probs=False,
        max_batch_size=8,
        # enable_chunked_context=True,
    )
    print(f"Loading model {model}")
    myexecutor = GenerationExecutor.create(engine_dir, config)
    sampling_params = SamplingParams(max_new_tokens=100)
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
        uvicorn.run(app, host="0.0.0.0", port=port)

    mpi_barrier()
    return True


async def generate_text_async(params: InputData):
    start_at = time.time()
    if params.request_type != 'CHAT':
        if len(params.prompt.split('Search query: ')) > 1:
            query = params.prompt.split('Search query: ')[1]
        else:
            query = "No query"
        prompt = tokenizer.encode(params.prompt)
    else:
        query = params.messages[1]['content'][14:]
        prompt = tokenizer.apply_chat_template(params.messages)
    sampling_params = SamplingParams(
        temperature=params.temperature,
        max_tokens=params.max_tokens,
        end_id=tokenizer.eos_token_id,
        stop_token_ids=[tokenizer.eos_token_id],
        seed=params.seed,
        # return_generation_logits=True,
        return_log_probs=True,
        top_k=0
    )
    stream = executor.generate_async(prompt,
                               sampling_params=sampling_params, streaming=True)
    responses = []
    first_at = None
    output_str = ""
    timeout = 10
    buffer = []
    async for output in stream:
        if first_at is None:
            first_at = time.time()
        token = tokenizer.decode(output.outputs[0].token_ids[-1])
        token_id = output.outputs[0].token_ids[-1]
        logprob = output.outputs[0].logprobs[-1]
        responses.append([token_id, logprob])
        buffer.append([token_id, logprob])
        if len(buffer) > BUFFER_SIZE:
            yield json.dumps(buffer)
            buffer = []
        
        output_str += token
        
        if timeout < time.time() - start_at:
            print(f"timeout {time.time() - start_at}s: {query}")
            break

    if len(buffer) > 0:
        yield json.dumps(buffer)
    # output_str = "".join(responses)
    tps = len(responses) / (time.time() - start_at)
    try:
        global tps_list
        tps_list.append(int(tps))
        average_tps = sum(tps_list[-20:]) / len(tps_list[-20:])
        first_average = sum(tps_list[0:20]) / len(tps_list[0:20])
        print(f"query: {query}")
        print(f"response: {output_str}")
        print(f"tps: {tps}, {len(responses)} tokens in {time.time() - start_at} seconds, first token: {first_at - start_at}")
        print(f"average tps: {average_tps}/{first_average} requests: {len(tps_list)}, {tps_list[-10:]}")
        global is_exit
        if len(tps_list) > 50 and (average_tps < 200 or average_tps < first_average * 0.8) and is_exit == False:
            is_exit = True
            print("average tps is too low, restarting the server")
            os._exit(1)

    except Exception as e:
        traceback.print_exc()

async def generate_text(messages, max_tokens, seed, timeout=10):
    output_str = ""
    async for output in generate_text_async(messages, max_tokens, seed, timeout):
        output_str += output
    return output_str

@app.post("/generate")
async def generate(data: InputData):
    output_tokens = {}
    async for output in generate_text_async(data):
        output_tokens = output
    return JSONResponse(content=output_tokens)

@app.post("/generate_async")
def generate_async(data: InputData):
    stream = generate_text_async(data)
    return StreamingResponse(stream, media_type="text/plain")

@app.get("/health")
def health():
    return "Healthy"

def parse_args():
    parser = argparse.ArgumentParser(description="Llama single model example")
    parser.add_argument(
        "-m", "--model",
        default="NousResearch/Meta-Llama-3.1-8B-Instruct",
    )
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
        '-p', "--port",
        type=int,
        default=9001,
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
                           [args.model] * args.tp_size,
                           [args.hf_model_dir] * args.tp_size,
                           [args.engine_dir] * args.tp_size,
                           [args.build] * args.tp_size,
                           [args.tp_size] * args.tp_size, 
                           [args.port] * args.tp_size,
                           range(args.tp_size))
        for r in results:
            assert r

if __name__ == "__main__":
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_dir)
    main(args)
