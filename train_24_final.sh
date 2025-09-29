#!/bin/bash
set -e

batch_size=1
ctx_len=4096
samples=128
lr=0.0001
lambda_param=0.001
lambda2_param=0.001

declare -A models=(
    ["Llama-3.2-1B"]="path/of/your/model"
)

dataset_path="path/of/your/data"
dataset_name=c4

results_file="ppl_results_$(date +%Y%m%d_%H%M%S).txt"
echo "Model | PPL" > $results_file
echo "--------------------" >> $results_file

cd searching

for model_name in "${!models[@]}"; do
    model_path=${models[$model_name]}
    echo ">>> Running $model_name"

    timestamp=$(date +%Y%m%d_%H%M%S)
    saving_path="/semi_structured24_alpha_0.3_${model_name}_${dataset_name}_${samples}_${lr}_${batch_size}_lambda${lambda_param}_${lambda2_param}_${timestamp}"
    record_path=$saving_path

    CUDA_VISIBLE_DEVICES=0 python main.py \
        --model_path "$model_path" \
        --dataset_path "$dataset_path" \
        --saving_path "$saving_path" \
        --ctx_len $ctx_len \
        --batch_size $batch_size \
        --samples $samples \
        --learning_rate $lr \
        --lambda_param $lambda_param \
        --lambda2_param $lambda2_param

    python extract_mask_mixed.py \
        --model_path "$model_path" \
        --output_path "$record_path" \
        --record_path "$record_path" \
        --mode 'semi' \
        --keep 2 \
        --group 4 \
        --use_gamma True

    sparse_model="$record_path/pruned_model_semi_gamma_only"

    ppl_output=$(python3 ppl.py \
        --model "$model_path" \
        --sparse_model "$sparse_model" \
        --ctx_len $ctx_len)

    echo "$model_name | $ppl_output" >> $results_file
done

echo "saving $results_file"
