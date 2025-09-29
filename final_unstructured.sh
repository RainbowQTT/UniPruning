#!/bin/bash
set -e

# 固定参数
batch_size=1
ctx_len=4096
lr=0.0001
lambda_param=0.001 # lambda1
lambda2_param=0.0 # l2
samples=128
dataset_path="path/of/your/data"
dataset_name=c4

declare -A models=(
  ["Llama-2-13B"]="path/of/your/model"
)

timestamp=$(date +%Y%m%d_%H%M%S)
report_file="report_${timestamp}.txt"
echo "Experiment Summary (${timestamp})" > $report_file
echo "ctx_len=${ctx_len}, lr=${lr}, samples=${samples}, lambda=${lambda_param}" >> $report_file
echo "-------------------------------------------------" >> $report_file

cd searching

for model_name in "${!models[@]}"; do
    model_path=${models[$model_name]}
    echo ">>> Running $model_name (lambda=${lambda_param}, samples=${samples})"

    saving_path="/unstructured_saving_128samples/${model_name}_${dataset_name}_${samples}_${lr}_${batch_size}_lambda${lambda_param}_${lambda2_param}_${timestamp}"
    record_path=$saving_path
    mkdir -p $saving_path

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py \
        --model_path "$model_path" \
        --dataset_path "$dataset_path" \
        --saving_path "$saving_path" \
        --ctx_len $ctx_len \
        --batch_size $batch_size \
        --samples $samples \
        --learning_rate $lr \
        --lambda_param $lambda_param \
        --lambda2_param $lambda2_param 

    ratio=0.6
    echo ">>> Extracting & Testing $model_name ratio=${ratio}"

    python extract_mask_mixed.py \
        --model_path "$model_path" \
        --output_path "$record_path" \
        --record_path "$record_path" \
        --mode 'unstructured' \
        --unstructured_ratio $ratio \
        --use_gamma True

    sparse_model="$record_path/pruned_model_unstructured_by_layer_${ratio}"
    ppl_out=$(python3 /ppl.py \
        --model "$model_path" \
        --sparse_model "$sparse_model" \
        --ctx_len $ctx_len)

    echo "model=${model_name}, lambda=${lambda_param}, samples=${samples}, ratio=${ratio}, ppl=${ppl_out}" >> ../$report_file
done

cd ..
echo "All experiments finished. Results saved in $report_file"
