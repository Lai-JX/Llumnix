#!/bin/bash
export HEAD_NODE_IP='127.0.0.1'

# $1: tp
# $2: 请求数量
# $3: 实例总数
# $4: 模型
# $5: 分布类型
TP=$1
REQ_NUM=$2
TOTAL_INSTANCES=$3
MODEL=$4
DISTRIBUTION=$5     # "burst", "uniform", "poisson", "gamma"
QPS=${6:-4}
MODEL_PATH="/share/models/llama-2-7b"

if [ "$MODEL" == "llama-2-7b" ]; then
    MODEL_PATH="/share/models/llama-2-7b"
elif [ "$MODEL" == "llama-2-13b" ]; then
    MODEL_PATH="/share/models/llama-2-13b"
elif [ "$MODEL" == "llama-7b" ]; then
    MODEL_PATH="/share/models/llama/llama-7b"
fi
echo "模型路径: $MODEL_PATH"

BASE_DIR='/workspace/llm-serve/Llumnix/logs/l40-pdd-'$TOTAL_INSTANCES/$MODEL/$DISTRIBUTION
mkdir -p $BASE_DIR

nvidia-smi -pl 300
nvidia-smi -pm ENABLED
nvidia-smi -acp 0

sm_clocks=("2490")
mem_clocks=("9001")

# 生成所有可能的 prefill 和 decode 组合
generate_instance_combinations() {
    local total=$1
    local combinations=()
    for ((prefill=1; prefill <= total - 1; prefill++)); do
        local decode=$((total - prefill))
        combinations+=("($prefill,$decode)")
    done
    echo "${combinations[@]}"
}

Llumnix_benchmark() {
    local count=$1
    HEAD_NODE=1 python -m llumnix.entrypoints.vllm.api_server \
                    --host 127.0.0.1 \
                    --port 1234 \
                    --initial-instances 1 \
                    --launch-ray-cluster \
                    --model $MODEL_PATH \
                    --worker-use-ray \
                    --migration-backend rayrpc \
                    --log-instance-info \
                    --tensor-parallel-size $TP \
                    --log-filename $BASE_DIR/serve_$count\_tp$TP\_$REQ_NUM\_qps_$QPS > $BASE_DIR/serve_$count\_tp$TP\_$REQ_NUM\_qps_$QPS.log 2>&1 &
    sleep 5
    # 启动实例
    count=$(($count - 1))
    for ((i=0; i < count; i++)); do
        port=$((1235 + $i))
        echo $port
        python -m llumnix.entrypoints.vllm.api_server \
                    --host 127.0.0.1 \
                    --port $port \
                    --initial-instances 1 \
                    --model $MODEL_PATH \
                    --worker-use-ray \
                    --migration-backend rayrpc \
                    --log-instance-info \
                    --tensor-parallel-size $TP \
                    --log-filename $BASE_DIR/serve_$((count + 1))\_tp$TP\_$REQ_NUM\_qps_$QPS > output2.log 2>&1 &
    done

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
        --dataset_type "sharegpt" \
        --dataset_path /workspace/llm-serve/sharegpt_gpt4.jsonl \
        --distribution $DISTRIBUTION \
        --log_latencies \
        --fail_on_response_failure \
        --max_request_len 2048 \
        --log_filename $BASE_DIR/benchmark_$((count + 1))\_tp$TP\_$REQ_NUM\_qps_$QPS \
        --prompt_save_path /workspace/llm-serve/Llumnix/logs/prompts/benchmark_$MODEL\_$DISTRIBUTION\_$REQ_NUM\_qps_$QPS \
        --qps $QPS

    # 关闭服务
    ./kill.sh
    sleep 5
}

Llumnix_benchmark_pdd() {
    local prefill_count=$1
    local decode_count=$2
    
    HEAD_NODE=1 python -m llumnix.entrypoints.vllm.api_server \
                --host 127.0.0.1 \
                --port 1234 \
                --initial-instances 1 \
                --launch-ray-cluster \
                --enable-pd-disagg --instance-type prefill \
                --model $MODEL_PATH \
                --worker-use-ray \
                --migration-backend rayrpc \
                --enable-migration \
                --log-instance-info \
                --tensor-parallel-size $TP \
                --log-filename $BASE_DIR/serve_pdd_tp$TP\_$REQ_NUM\_qps_$QPS\_$prefill_count\_$decode_count > $BASE_DIR/serve_pdd_tp$TP\_$REQ_NUM\_qps_$QPS\_$prefill_count\_$decode_count.log 2>&1 &
    sleep 5
    # 启动 prefill 实例
    prefill_count=$(($prefill_count - 1))
    for ((i=0; i < prefill_count; i++)); do
        port=$((1235 + $i))
        echo $port
        python -m llumnix.entrypoints.vllm.api_server \
                    --host 127.0.0.1 \
                    --port $port \
                    --initial-instances 1 \
                    --enable-pd-disagg --instance-type prefill \
                    --model $MODEL_PATH \
                    --worker-use-ray \
                    --migration-backend rayrpc \
                    --enable-migration \
                    --log-instance-info \
                    --tensor-parallel-size $TP \
                    --log-filename $BASE_DIR/serve_pdd_tp$TP\_$REQ_NUM\_qps_$QPS\_$((prefill_count + 1))\_$decode_count > output1.log 2>&1 &
    done

    # 启动 decode 实例
    for ((i=0; i < decode_count; i++)); do
        port=$((1245 + $i))
        echo $port
        sleep 1
        python -m llumnix.entrypoints.vllm.api_server \
                    --host 127.0.0.1 \
                    --port $port \
                    --initial-instances 1 \
                    --enable-pd-disagg --instance-type decode \
                    --model $MODEL_PATH \
                    --worker-use-ray \
                    --migration-backend rayrpc \
                    --enable-migration \
                    --log-instance-info \
                    --tensor-parallel-size $TP \
                    --log-filename $BASE_DIR/serve_pdd_tp$TP\_$REQ_NUM\_qps_$QPS\_$((prefill_count + 1))\_$decode_count > output2.log 2>&1 &
    done

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
        --dataset_type "sharegpt" \
        --dataset_path /workspace/llm-serve/sharegpt_gpt4.jsonl \
        --distribution $DISTRIBUTION \
        --log_latencies \
        --fail_on_response_failure \
        --max_request_len 2048 \
        --log_filename $BASE_DIR/benchmark_pdd_tp$TP\_$REQ_NUM\_qps_$QPS\_$((prefill_count + 1))\_$decode_count \
        --prompt_save_path /workspace/llm-serve/Llumnix/logs/prompts/benchmark_$MODEL\_$DISTRIBUTION\_$REQ_NUM\_qps_$QPS \
        --qps $QPS

    # 关闭服务
    ./kill.sh
    sleep 5
}

for sm_clock in "${sm_clocks[@]}"; do
    for mem_clock in "${mem_clocks[@]}"; do
        echo "clock : $sm_clock  $mem_clock"    # 暂时没用

        Llumnix_benchmark $TOTAL_INSTANCES

        # 获取所有可能的实例组合
        combinations=$(generate_instance_combinations $TOTAL_INSTANCES)
        # combinations=('(1,3)')
        echo $combinations
        for combo in $combinations; do
            echo $combo
            # 使用 tr 命令删除括号，然后用 awk 提取两个数字
            prefill_count=$(echo $combo | tr -d '()' | awk -F, '{print $1}')
            decode_count=$(echo $combo | tr -d '()' | awk -F, '{print $2}')
        
            

            Llumnix_benchmark_pdd $prefill_count $decode_count
        done

        nvidia-smi -acp 1
        nvidia-smi -rac
        nvidia-smi -rgc
        nvidia-smi -rmc
    done
done

nvidia-smi -acp 1
nvidia-smi -rac
nvidia-smi -rgc
nvidia-smi -rmc