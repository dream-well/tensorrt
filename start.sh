source /workspace/venv/bin/activate
export HF_HOME=/workspace/hf
export HF_DATASETS_CACHE=/workspace/hf
MODEL=${1:-NousResearch/Meta-Llama-3.1-8B-Instruct}
PORT=${2:-9001}
export HF_LLAMA_MODEL=$(python3 -c "from pathlib import Path; from huggingface_hub import hf_hub_download; print(Path(hf_hub_download('$MODEL', filename='config.json')).parent)")
export TRT_ENGINES_DIR=/workspace/trt_engines/engines/$MODEL
export TRT_CHECKPOINT_DIR=/workspace/trt_engines/checkpoints/$MODEL
python3 serve.py --hf_model_dir $HF_LLAMA_MODEL --engine_dir $TRT_ENGINES_DIR --tp_size 8 --port $PORT --model $MODEL
