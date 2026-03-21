# !/bin/bash

export CUDA_VISIBLE_DEVICES=0

CKPT_PATH="/inspire/hdd/project/exploration-topic/public/ent/TIFO/ckpt/t2i_generation1/14192567000.0/ckpt/iter_999.pth"
PROMPT="A beautiful sunset over a futuristic city, high resolution, digital art"

# 5. 运行 Python 脚本
python /inspire/hdd/project/exploration-topic/public/ent/TIFO/generation_inference.py \
    --ckpt_path "$CKPT_PATH" \
    --prompt "$PROMPT"

echo "生成任务已完成，请检查 generated_samples 目录。"