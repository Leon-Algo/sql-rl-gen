#!/usr/bin/env bash
set -e

# 如果未在外部激活，这里确保进入项目虚拟环境
if [ -f /opt/conda/bin/activate ]; then
       # shellcheck disable=SC1091
       source /opt/conda/bin/activate sql-rl-gen || true
fi

# python sql_rl_gen/generation/sql_generation.py \
#        --model_name_or_path "juierror/flan-t5-text2sql-with-schema-v2" \
#        --dataset "example_text2sql_$1_train" \
#        --template llama3 \
#        --outdir "./output/model_$1_train" \
#        --steps_n 1000 \
#        --dataset_name $1

python sql_rl_gen/generation/sql_generation.py \
       --model_name_or_path "./local_models/flan-t5-text2sql-with-schema-v2-unpacked" \
       --dataset "example_text2sql_spider_train" \
       --template llama3 \
       --outdir "./output/model_spider_train" \
       --steps_n 1000 \
       --dataset_name spider