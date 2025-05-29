#!/bin/bash
export HEAD_NODE_IP='127.0.0.1'
export RAY_DEDUP_LOGS=0 

# 解析参数
MODEL=${1:-"llama-2-7b"}
QPS=${2:-4}
DATASET=${3:-"/workspace/llm-serve/sharegpt_gpt4.jsonl"}      # "burst", "uniform", "poisson", "gamma"
TP=${4:-2}
INSTANCE_NUM=${5:-1}
REQ_NUM=${6:-20}
MIGRATION_BACKEND=${7:-"rayrpc"}

MODEL_PATH="/share/models/llama-2-7b"


if [ "$MODEL" == "llama-2-7b" ]; then
    MODEL_PATH="/share/models/llama-2-7b"
elif [ "$MODEL" == "llama-2-13b" ]; then
    MODEL_PATH="/share/models/llama-2-13b"
elif [ "$MODEL" == "llama-7b" ]; then
    MODEL_PATH="/share/models/llama/llama-7b"
fi
echo "模型路径: $MODEL_PATH"

BASE_DIR='/workspace/llm-serve/Llumnix/logs/a6000-osdi24ae'/$MODEL/$MIGRATION_BACKEND
mkdir -p $BASE_DIR

# nvidia-smi -pl 300
nvidia-smi -pm ENABLED
nvidia-smi -acp 0
# rm -r /root/.cache/bazel/_bazel_root/
# 启动prefill实例
HEAD_NODE=1 python -u /workspace/llm-serve/Llumnix/vllm/entrypoints/api_server.py \
            --host 127.0.0.1 \
            --port 1234 \
            --initial-instances $INSTANCE_NUM \
            --launch-ray-cluster \
            --model $MODEL_PATH \
            --worker-use-ray \
            --migration-backend $MIGRATION_BACKEND \
            --enable-migration \
            --log-instance-info \
            --log-request-timestamps \
            --tensor-parallel-size $TP \
            --max-num-seqs $REQ_NUM \
            --migration-num-layers 32 \
            --log-filename $BASE_DIR/serve_tp$TP\_n$INSTANCE_NUM\_$REQ_NUM\_qps_$QPS > $BASE_DIR/serve_tp$TP\_n$INSTANCE_NUM\_$REQ_NUM\_qps_$QPS.log 2>&1 &
sleep 10

# HEAD_NODE=1 python -m llumnix.entrypoints.vllm.api_server \
#                 --host 127.0.0.1 \
#                 --port 1234 \
#                 --initial-instances 1 \
#                 --launch-ray-cluster \
#                 --tensor-parallel-size 4 \
#                 --model $MODEL_PATH \
#                 --worker-use-ray \
#                 --migration-backend gloo \
# 判断$HEAD_NODE_IP:1234是否可用 
INTERVAL=2
while true; do
    if curl -s -o /dev/null -I http://$HEAD_NODE_IP:1234; then
        echo "Port 1234 on $HEAD_NODE_IP is open and reachable."
        sleep $INTERVAL
        break
    else
        echo "Port 1234 on $HEAD_NODE_IP is not reachable. Retrying in $INTERVAL seconds..."
        sleep $INTERVAL
    fi
done

# 添加负载
# python /workspace/llm-serve/Llumnix/benchmark/benchmark_serving.py \
#     --ip_ports $HEAD_NODE_IP:1234 \
#     --tokenizer $MODEL_PATH \
#     --random_prompt_count $REQ_NUM \
#     --gen_random_prompts \
#     --random_prompt_lens_mean $prompt_len \
#     --random_prompt_lens_range 0 \
#     --variable_prompt_lens_distribution "uniform" \
#     --allow_variable_generation_length \
#     --variable_response_lens_mean $response_len \
#     --variable_response_lens_range 0 \
#     --variable_response_lens_distribution "uniform" \
#     --distribution $DISTRIBUTION \
#     --log_latencies \
#     --fail_on_response_failure \
#     --log_filename $BASE_DIR/benchmark_pdd_tp$prefill_tp\_n$prefill_count\_tp$decode_tp\_n$decode_count\_$REQ_NUM\_qps_$QPS\_prompt_len_$prompt_len\_response_len_$response_len \
#     --prompt_save_path /workspace/llm-serve/Llumnix/logs/prompts/benchmark_$MODEL\_$DISTRIBUTION\_$REQ_NUM\_qps_$QPS\_prompt_len_$prompt_len\_response_len_$response_len \
#     --qps $QPS
python /workspace/llm-serve/Llumnix/benchmark/benchmark_serving.py \
    --host $HEAD_NODE_IP \
    --port 1234 \
    --tokenizer $MODEL_PATH \
    --num-prompts $REQ_NUM \
    --dataset /workspace/llm-serve/sharegpt_gpt4.jsonl \
    --prompt_save_path /workspace/llm-serve/Llumnix/logs/prompts/benchmark_$MODEL\_$REQ_NUM\_sharegpt \
    --request-rate $QPS

sleep 5
./kill.sh
