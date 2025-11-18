#!/usr/bin/env bash
set -e

# 如果未在外部激活，这里确保进入项目虚拟环境
if [ -f /opt/conda/bin/activate ]; then
  # shellcheck disable=SC1091
  source /opt/conda/bin/activate sql-rl-gen || true
fi

python scripts/run_compare_eval.py \
  --model_name_or_path ./local_models/flan-t5-text2sql-with-schema-v2-unpacked \
  --trained_agent_path output/model_spider_train/best \
  --dataset_name spider \
  --template llama3 \
  --out_md output/compare_spider_base_vs_rl.md