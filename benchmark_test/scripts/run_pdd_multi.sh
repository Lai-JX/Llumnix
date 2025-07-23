#!/bin/bash
export HEAD_NODE_IP='127.0.0.1'
export RAY_DEDUP_LOGS=0 
# $1: tp
# $2: 请求数量
# $3: 实例总数
# $4: 模型
# $5: 分布类型
# $6: qps
# $7: gpu类型
# $8: 最大迁移并发数
TP=$1
REQ_NUM=$2
TOTAL_INSTANCES=$3
MODEL=$4
DISTRIBUTION=$5     # "burst", "uniform", "poisson", "gamma"
QPS=${6:-4}
MODEL_PATH="/share/models/llama-2-7b"
gputype=${7:-'A6000-t2'}
max_migration_concurrency=${8:-1}

if [ "$MODEL" == "llama-2-7b" ]; then
    MODEL_PATH="/share/models/llama-2-7b"
    max_request_len=2048
elif [ "$MODEL" == "llama-2-13b" ]; then
    MODEL_PATH="/share/models/llama-2-13b"
    max_request_len=4096
elif [ "$MODEL" == "llama-7b" ]; then
    MODEL_PATH="/share/models/llama/llama-7b"
    max_request_len=2048
elif [ "$MODEL" == "llama-13b" ]; then
    MODEL_PATH="/share/models/llama/llama-13b"
    max_request_len=4096
elif [ "$MODEL" == "llama-30b" ]; then
    MODEL_PATH="/share/models/llama/llama-30b"
    max_request_len=8192
fi
echo "模型路径: $MODEL_PATH"

BASE_DIR="/workspace/llm-serve/Llumnix/benchmark_test/logs/$gputype-pdd-"$TOTAL_INSTANCES/$MODEL/$DISTRIBUTION
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
    # return
    local count=$1
    filename=$BASE_DIR/benchmark_$((count))\_tp$TP\_$REQ_NUM\_qps_$QPS\_latency_info.json
    if [ -e $filename ]; then
        echo "Llumnix_benchmark already test"
        return
    fi
    rm $BASE_DIR/serve_$count\_tp$TP\_$REQ_NUM\_qps_$QPS.log
    rm $BASE_DIR/serve_$count\_tp$TP\_$REQ_NUM\_qps_$QPS\_instance.csv
    port_base=1234
    HEAD_NODE=1 python -u -m llumnix.entrypoints.vllm.api_server \
                    --host 127.0.0.1 \
                    --port $port_base \
                    --initial-instances 1 \
                    --launch-ray-cluster \
                    --model $MODEL_PATH \
                    --worker-use-ray \
                    --enable-migration \
                    --migration-backend rayrpc \
                    --log-instance-info \
                    --log-request-timestamps \
                    --tensor-parallel-size $TP \
                    --request-output-queue-type zmq \
                    --max-migration-concurrency $max_migration_concurrency \
                    --log-filename $BASE_DIR/serve_$count\_tp$TP\_$REQ_NUM\_qps_$QPS > $BASE_DIR/serve_$count\_tp$TP\_$REQ_NUM\_qps_$QPS.log 2>&1 &
                    #--request-output-queue-port $(($port_base + 50)) \
    echo $(($port_base + 50))

    sleep 15
    # 启动实例
    port_base=$(($port_base + 1))
    count=$(($count - 1))
    for ((i=0; i < count; i++)); do
        port=$(($port_base + $i))
        echo $port
        python -u -m llumnix.entrypoints.vllm.api_server \
                    --host 127.0.0.1 \
                    --port $port \
                    --initial-instances 1 \
                    --model $MODEL_PATH \
                    --worker-use-ray \
                    --enable-migration \
                    --migration-backend rayrpc \
                    --log-instance-info \
                    --log-request-timestamps \
                    --tensor-parallel-size $TP \
                    --request-output-queue-type zmq \
                    --max-migration-concurrency $max_migration_concurrency \
                    --log-filename $BASE_DIR/serve_$((count + 1))\_tp$TP\_$REQ_NUM\_qps_$QPS > output2.log 2>&1 &
                    #--request-output-queue-port $(($port_base + 50)) \
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

    # 构造所有端口的ip_ports参数
    ip_ports="$HEAD_NODE_IP:1234"
    for ((i=0; i < count; i++)); do
        ip_ports="$ip_ports $HEAD_NODE_IP:$((1235 + i))"
    done
    echo "ip_ports: $ip_ports"

    # 添加负载
    python -u /workspace/llm-serve/Llumnix/benchmark/benchmark_serving.py \
        --ip_ports $ip_ports \
        --tokenizer $MODEL_PATH \
        --random_prompt_count $REQ_NUM \
        --dataset_type "sharegpt" \
        --dataset_path /workspace/llm-serve/sharegpt_gpt4.jsonl \
        --distribution $DISTRIBUTION \
        --log_latencies \
        --fail_on_response_failure \
        --max_request_len $max_request_len \
        --log_filename $BASE_DIR/benchmark_$((count + 1))\_tp$TP\_$REQ_NUM\_qps_$QPS. \
        --prompt_save_path /workspace/llm-serve/Llumnix/benchmark_test/logs/prompts/sharegpt_$MODEL\_$DISTRIBUTION\_$REQ_NUM\_qps_$QPS \
        --qps $QPS 2>&1 | tee -a $BASE_DIR/serve_$((count + 1))\_tp$TP\_$REQ_NUM\_qps_$QPS\_benchmark.log

    # 关闭服务
    ./kill.sh
    sleep 5
}

Llumnix_benchmark_pdd() {
    local prefill_count=$1
    local decode_count=$2
    filename=$BASE_DIR/benchmark_pdd_tp$TP\_$REQ_NUM\_qps_$QPS\_$((prefill_count))\_$decode_count\_latency_info.json
    if [ -e $filename ]; then
        echo "Llumnix_benchmark_pdd already test"
        return
    fi
    rm $BASE_DIR/serve_pdd_tp$TP\_$REQ_NUM\_qps_$QPS\_$prefill_count\_$decode_count\_instance.csv
    rm $BASE_DIR/serve_pdd_tp$TP\_$REQ_NUM\_qps_$QPS\_$prefill_count\_$decode_count.log
    port_base=1234
    HEAD_NODE=1 python -u -m llumnix.entrypoints.vllm.api_server \
                --host 127.0.0.1 \
                --port $port_base \
                --initial-instances 1 \
                --launch-ray-cluster \
                --enable-pd-disagg --instance-type prefill \
                --model $MODEL_PATH \
                --worker-use-ray \
                --migration-backend rayrpc \
                --enable-migration \
                --log-instance-info \
                --log-request-timestamps \
                --tensor-parallel-size $TP \
                --request-output-queue-type zmq \
                --max-migration-concurrency $max_migration_concurrency \
                --log-filename $BASE_DIR/serve_pdd_tp$TP\_$REQ_NUM\_qps_$QPS\_$prefill_count\_$decode_count > $BASE_DIR/serve_pdd_tp$TP\_$REQ_NUM\_qps_$QPS\_$prefill_count\_$decode_count.log 2>&1 &
                #--request-output-queue-port $(($port_base + 50)) \
                #--max-num-seqs $REQ_NUM \

    sleep 15
    # 启动 prefill 实例
    port_base=$(($port_base + 1))
    prefill_count=$(($prefill_count - 1))
    for ((i=0; i < prefill_count; i++)); do
        port=$(($port_base + $i))
        echo $port
        python -u -m llumnix.entrypoints.vllm.api_server \
                    --host 127.0.0.1 \
                    --port $port \
                    --initial-instances 1 \
                    --enable-pd-disagg --instance-type prefill \
                    --model $MODEL_PATH \
                    --worker-use-ray \
                    --migration-backend rayrpc \
                    --enable-migration \
                    --log-instance-info \
                    --log-request-timestamps \
                    --tensor-parallel-size $TP \                    
                    --request-output-queue-type zmq \
                    --max-migration-concurrency $max_migration_concurrency \
                    --log-filename $BASE_DIR/serve_pdd_tp$TP\_$REQ_NUM\_qps_$QPS\_$((prefill_count + 1))\_$decode_count > output1.log 2>&1 &
                    #--request-output-queue-port $(($port_base + 50)) \
                    #--max-num-seqs $REQ_NUM \
    done
    # sleep 10
    # 启动 decode 实例
    port_base=$(($port_base + $prefill_count))
    for ((i=0; i < decode_count; i++)); do
        port=$(($port_base + $i))
        echo $port
        sleep 1
        python -u -m llumnix.entrypoints.vllm.api_server \
                    --host 127.0.0.1 \
                    --port $port \
                    --initial-instances 1 \
                    --enable-pd-disagg --instance-type decode \
                    --model $MODEL_PATH \
                    --worker-use-ray \
                    --migration-backend rayrpc \
                    --enable-migration \
                    --log-instance-info \
                    --log-request-timestamps \
                    --tensor-parallel-size $TP \
                    --request-output-queue-type zmq \
                    --max-migration-concurrency $max_migration_concurrency \
                    --log-filename $BASE_DIR/serve_pdd_tp$TP\_$REQ_NUM\_qps_$QPS\_$((prefill_count + 1))\_$decode_count > output2.log 2>&1 &
                    #--request-output-queue-port $(($port + 50)) \
                    #--max-num-seqs $REQ_NUM \
                    
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

    # 构造所有 prefill 和 decode 实例端口的ip_ports参数
    ip_ports="$HEAD_NODE_IP:1234"
    for ((i=0; i < prefill_count; i++)); do
        ip_ports="$ip_ports $HEAD_NODE_IP:$((1235 + i))"
    done
    for ((i=0; i < decode_count; i++)); do
        ip_ports="$ip_ports $HEAD_NODE_IP:$(($port_base + i))"
    done
    echo "ip_ports: $ip_ports"

    # 添加负载
    python -u /workspace/llm-serve/Llumnix/benchmark/benchmark_serving.py \
        --ip_ports $ip_ports \
        --tokenizer $MODEL_PATH \
        --random_prompt_count $REQ_NUM \
        --dataset_type "sharegpt" \
        --dataset_path /workspace/llm-serve/sharegpt_gpt4.jsonl \
        --distribution $DISTRIBUTION \
        --log_latencies \
        --fail_on_response_failure \
        --max_request_len $max_request_len \
        --log_filename $BASE_DIR/benchmark_pdd_tp$TP\_$REQ_NUM\_qps_$QPS\_$((prefill_count + 1))\_$decode_count. \
        --prompt_save_path /workspace/llm-serve/Llumnix/benchmark_test/logs/prompts/sharegpt_$MODEL\_$DISTRIBUTION\_$REQ_NUM\_qps_$QPS \
        --qps $QPS  2>&1 | tee -a $BASE_DIR/serve_pdd_tp$TP\_$REQ_NUM\_qps_$QPS\_$((prefill_count + 1))\_$decode_count\_benchmark.log

    # 关闭服务
    ./kill.sh
    sleep 15
}
echo "参数: TP=$TP, REQ_NUM=$REQ_NUM, TOTAL_INSTANCES=$TOTAL_INSTANCES, MODEL=$MODEL, DISTRIBUTION=$DISTRIBUTION, QPS=$QPS, MODEL_PATH=$MODEL_PATH, BASE_DIR=$BASE_DIR"
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