pm2 start start.sh --name s9001 -- NousResearch/Meta-Llama-3.1-8B-Instruct 9001 &
pm2 start start.sh --name s9002 -- NousResearch/Hermes-3-Llama-3.1-8B 9002 &
pm2 start start.sh --name s9003 -- NTQAI/Nxcode-CQ-7B-orpo 9003 &
pm2 start start.sh --name s9004 -- gryphe/mythomax-l2-13b 9004 &
# pm2 start start.sh --name s9005 -- deepseek-ai/deepseek-coder-33b-instruct 9005 &