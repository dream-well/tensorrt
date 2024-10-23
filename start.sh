source /workspace/venv/bin/activate
export HF_HOME=/workspace/hf
export HF_DATASETS_CACHE=/workspace/hf
export HF_HOME=/workspace/hf
export HF_DATASETS_CACHE=/workspace/hf
export HF_LLAMA_MODEL=`python3 -c "from pathlib import Path; from huggingface_hub import hf_hub_download; print(Path(hf_hub_download('NousResearch/Meta-Llama-3.1-8B-Instruct', filename='config.json')).parent)"`
export TRT_ENGINES_DIR=/workspace/trt_engines/engines
export TRT_CHECKPOINT_DIR=/workspace/trt_engines/checkpoints
python3 serve.py --hf_model_dir $HF_LLAMA_MODEL --engine_dir $TRT_ENGINES_DIR --tp_size 8 
