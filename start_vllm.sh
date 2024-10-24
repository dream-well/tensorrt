source /workspace/venv/bin/activate
export HF_HOME=/workspace/hf
export HF_DATASETS_CACHE=/workspace/hf
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export NCCL_P2P_DISABLE=1
# export OMP_NUM_THREADS=48
python3 serve_vllm.py
