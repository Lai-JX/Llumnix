#!/bin/bash
export HEAD_NODE_IP='127.0.0.1'

# prefill 1*tp1,
# decode 1*tp2,


REQ_NUM=10
TOTAL_INSTANCES=$3
MODEL='llama-2-7b'
DISTRIBUTION='uniform'     # "burst", "uniform", "poisson", "gamma"
QPS=${6:-4}
MIGRATION_BACKEND='gloo'        # rayrpc gloo nccl
MODEL_PATH="/share/models/llama-2-7b"
prefill_tp=1
decode_tp=2
prefill_count=1
decode_count=1
prompt_len=256
response_len=128

if [ "$MODEL" == "llama-2-7b" ]; then
    MODEL_PATH="/share/models/llama-2-7b"
elif [ "$MODEL" == "llama-2-13b" ]; then
    MODEL_PATH="/share/models/llama-2-13b"
elif [ "$MODEL" == "llama-7b" ]; then
    MODEL_PATH="/share/models/llama/llama-7b"
fi
echo "模型路径: $MODEL_PATH"

BASE_DIR='/workspace/llm-serve/Llumnix/logs/l40-pdd--test'/$MODEL/$DISTRIBUTION/$MIGRATION_BACKEND
mkdir -p $BASE_DIR

nvidia-smi -pl 300
nvidia-smi -pm ENABLED
nvidia-smi -acp 0

# 启动prefill实例
HEAD_NODE=1 python -m llumnix.entrypoints.vllm.api_server \
            --host 127.0.0.1 \
            --port 1234 \
            --initial-instances 1 \
            --launch-ray-cluster \
            --enable-pd-disagg --instance-type prefill \
            --model $MODEL_PATH \
            --worker-use-ray \
            --migration-backend $MIGRATION_BACKEND \
            --enable-migration \
            --log-instance-info \
            --log-request-timestamps \
            --tensor-parallel-size $prefill_tp \
            --max-num-seqs $REQ_NUM \
            --log-filename $BASE_DIR/serve_pdd_tp$prefill_tp\_$REQ_NUM\_qps_$QPS\_$prefill_count\_$decode_count\_prompt_len_$prompt_len\_response_len_$response_len > $BASE_DIR/serve_pdd_tp$prefill_tp\_$REQ_NUM\_qps_$QPS\_$prefill_count\_$decode_count\_prompt_len_$prompt_len\_response_len_$response_len.log 2>&1 &

# 启动decode实例
python -m llumnix.entrypoints.vllm.api_server \
            --host 127.0.0.1 \
            --port 1235 \
            --initial-instances 1 \
            --enable-pd-disagg --instance-type decode \
            --model $MODEL_PATH \
            --worker-use-ray \
            --migration-backend $MIGRATION_BACKEND \
            --enable-migration \
            --log-instance-info \
            --log-request-timestamps \
            --tensor-parallel-size $decode_tp \
            --max-num-seqs $REQ_NUM \
            --log-filename $BASE_DIR/serve_pdd_tp$decode_tp\_$REQ_NUM\_qps_$QPS\_$((prefill_count + 1))\_$decode_count\_prompt_len_$prompt_len\_response_len_$response_len > output2.log 2>&1 &

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
python /workspace/llm-serve/Llumnix/benchmark/benchmark_serving.py \
    --ip_ports $HEAD_NODE_IP:1234 \
    --tokenizer $MODEL_PATH \
    --random_prompt_count $REQ_NUM \
    --gen_random_prompts \
    --random_prompt_lens_mean $prompt_len \
    --random_prompt_lens_range 0 \
    --variable_prompt_lens_distribution "uniform" \
    --allow_variable_generation_length \
    --variable_response_lens_mean $response_len \
    --variable_response_lens_range 0 \
    --variable_response_lens_distribution "uniform" \
    --distribution $DISTRIBUTION \
    --log_latencies \
    --fail_on_response_failure \
    --log_filename $BASE_DIR/benchmark_pdd_tp$prefill_tp\_$REQ_NUM\_qps_$QPS\_prompt_len_$prompt_len\_response_len_$response_len\_$((prefill_count + 1))\_$decode_count \
    --prompt_save_path /workspace/llm-serve/Llumnix/logs/prompts/benchmark_$MODEL\_$DISTRIBUTION\_$REQ_NUM\_qps_$QPS\_prompt_len_$prompt_len\_response_len_$response_len \
    --qps $QPS
