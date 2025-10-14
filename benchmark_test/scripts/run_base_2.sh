#!/bin/bash
export HEAD_NODE_IP='127.0.0.1'
# export RAY_DEDUP_LOGS=0 
# export VLLM_ATTENTION_BACKEND=FLASHINFER
# $1: prefill_tps，如"1,1"
# $2: decode_tps，如"2"
# $3: tps，如"2"
# $4: 请求数量
# $5: 模型
# $6: 分布类型
# $7: qps
# $8: log_dir_prefix
# $9: 最大迁移并发数
# $10: prompt_len            为空时采用 sharegpt 数据集
# $11: response_len

# 输出help
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage: $0 [PREFILL_TPS_STR] [DECODE_TPS_STR] [TPS_STR] REQ_NUM MODEL DISTRIBUTION [QPS] [log_dir_prefix] [max_migration_concurrency] [prompt_len] [response_len]"
    echo "Example: $0 \"1,1\" \"2\" 100 llama-7b uniform 4 l40-pdd-hetero 1 128 128"
    exit 0
fi

PREFILL_TPS_STR=${1:-"1,1"}
DECODE_TPS_STR=${2:-""}
TPS_STR=${3:-""}

IFS=',' read -ra PREFILL_TPS <<< "$PREFILL_TPS_STR"
IFS=',' read -ra DECODE_TPS <<< "$DECODE_TPS_STR"
IFS=',' read -ra TPS <<< "$DECODE_TPS_STR"

REQ_NUM=$4
MODEL=$5
DISTRIBUTION=$6     # "burst", "uniform", "poisson", "gamma"
QPS=${7:-4}
log_dir_prefix=${8:-"l40-pdd-hetero"}  # 默认值为 "l40-pdd-hetero"
max_migration_concurrency=${9:-1}
prompt_len=${10:-""}
response_len=${11:-""}

echo "PREFILL_TPS_STR: $PREFILL_TPS_STR"
echo "DECODE_TPS_STR: $DECODE_TPS_STR"
echo "TPS_STR: $TPS_STR"
echo "REQ_NUM: $REQ_NUM"
echo "MODEL: $MODEL"
echo "DISTRIBUTION: $DISTRIBUTION"
echo "QPS: $QPS"
echo "log_dir_prefix: $log_dir_prefix"

MODEL_PATH="/share/models/llama-2-7b"
MIGRATION_BACKEND="rayrpc"       # rayrpc gloo nccl

if [ "$MODEL" == "llama-2-7b" ]; then
    MODEL_PATH="/share/models/llama-2-7b"
    max_request_len=-1 # 2048 让benchmark_serving.py自动获取
elif [ "$MODEL" == "llama-2-13b" ]; then
    MODEL_PATH="/share/models/llama-2-13b"
    max_request_len=-1 # 4096 让benchmark_serving.py自动获取
elif [ "$MODEL" == "llama-7b" ]; then
    MODEL_PATH="/share/models/llama/llama-7b"
    max_request_len=-1 # 2048 让benchmark_serving.py自动获取
elif [ "$MODEL" == "llama-13b" ]; then
    MODEL_PATH="/share/models/llama/llama-13b"
    max_request_len=-1 # 4096 让benchmark_serving.py自动获取
fi
echo "模型路径: $MODEL_PATH"

BASE_DIR='/workspace/llm-serve/Llumnix/benchmark_test/logs/'$log_dir_prefix/$MODEL/$DISTRIBUTION
mkdir -p $BASE_DIR
filename_base=$BASE_DIR/benchmark\_$REQ_NUM\_qps-$QPS\_$PREFILL_TPS_STR-$DECODE_TPS_STR-$TPS_STR

# nvidia-smi -pl 300
nvidia-smi -pm ENABLED
nvidia-smi -acp 0

sm_clocks=("2490")
mem_clocks=("9001")
ip_ports=""
ip_ports_tmp=""

Llumnix_benchmark() {
    # return
    local count=$1
    local TP=$2
    port_base=1334

    # 启动实例
    port_base=$port_base
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
                    --migration-backend $MIGRATION_BACKEND \
                    --log-instance-info \
                    --log-request-timestamps \
                    --tensor-parallel-size $TP \
                    --request-output-queue-type zmq \
                    --max-migration-concurrency $max_migration_concurrency \
                    --log-filename $filename_base > output2.log 2>&1 &
                    #--request-output-queue-port $(($port_base + 50)) \
    done

    # 构造所有端口的ip_ports参数
    ip_ports_tmp=""
    for ((i=0; i < count; i++)); do
        ip_ports_tmp="$ip_ports_tmp $HEAD_NODE_IP:$(($port_base + i))"
    done
    echo "Llumnix_benchmark ip_ports: $ip_ports_tmp"
}

Llumnix_benchmark_pdd() {
    filename=$filename_base\_latency_info.json
    if [ -e $filename ]; then
        echo "Llumnix_benchmark_pdd($filename) already test"
        return
    fi
    rm $filename_base.log
    rm $filename_base\_instance.csv
    # 遍历 prefill_dps 和 decode_dps
    TP=0
    count=0
    for prefill_tp in "${PREFILL_TPS[@]}"; do
        TP=$prefill_tp
        count=$((count + 1))
        echo "PREFILL NO.$count, TP: $TP"
        # 判断是不是第一个实例
        if [ $count -eq 1 ]; then
            HEAD_NODE=1 python -u -m llumnix.entrypoints.vllm.api_server \
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
                --tensor-parallel-size $TP \
                --request-output-queue-type zmq \
                --max-num-seqs 512 \
                --max-migration-concurrency $max_migration_concurrency \
                --log-filename $filename_base > $filename_base.log 2>&1 &
                # --max-num-seqs $REQ_NUM \
                # --disable-async-output-proc \
                # --kv-cache-dtype 'fp8' \
            sleep 15
        else
            port=$((1233 + $count))
            echo $port
            python -u -m llumnix.entrypoints.vllm.api_server \
                --host 127.0.0.1 \
                --port $port \
                --initial-instances 1 \
                --enable-pd-disagg --instance-type prefill \
                --model $MODEL_PATH \
                --worker-use-ray \
                --migration-backend $MIGRATION_BACKEND \
                --enable-migration \
                --log-instance-info \
                --log-request-timestamps \
                --tensor-parallel-size $TP \
                --request-output-queue-type zmq \
                --max-num-seqs 512 \
                --max-migration-concurrency $max_migration_concurrency \
                --log-filename $filename_base > output1.log 2>&1 &
                # --max-num-seqs $REQ_NUM \
                # --disable-async-output-proc \
                # --kv-cache-dtype 'fp8' \
        fi
    done

    # 启动 decode 实例
    count=0
    for decode_dp in "${DECODE_TPS[@]}"; do
        TP=$decode_dp
        count=$((count + 1))
        port=$((1243 + $count))
        echo $port
        echo "DECODE NO.$count, TP: $TP"
        sleep 1
        python -u -m llumnix.entrypoints.vllm.api_server \
            --host 127.0.0.1 \
            --port $port \
            --initial-instances 1 \
            --enable-pd-disagg --instance-type decode \
            --model $MODEL_PATH \
            --worker-use-ray \
            --migration-backend $MIGRATION_BACKEND \
            --enable-migration \
            --log-instance-info \
            --log-request-timestamps \
            --tensor-parallel-size $TP \
            --request-output-queue-type zmq \
            --max-num-seqs 512 \
            --max-migration-concurrency $max_migration_concurrency \
            --log-filename $filename_base > output2.log 2>&1 &
            # --max-num-seqs $REQ_NUM \
            # --kv-cache-dtype 'fp8' \
        
    done

    # 构造所有 prefill 和 decode 实例端口的ip_ports参数
    ip_ports_tmp="$HEAD_NODE_IP:1234"
    for ((i=1; i < ${#PREFILL_TPS[@]}; i++)); do
        ip_ports_tmp="$ip_ports_tmp $HEAD_NODE_IP:$((1234 + i))"
    done
    ip_ports_tmp="$ip_ports_tmp $HEAD_NODE_IP:1244"
    for ((i=1; i < ${#DECODE_TPS[@]}; i++)); do
        ip_ports_tmp="$ip_ports_tmp $HEAD_NODE_IP:$((1244 + i))"
    done
    echo "Llumnix_benchmark_pdd ip_ports: $ip_ports_tmp"
}

add_workload(){
    echo "All ip_ports: $ip_ports"
    # 判断是否所有端口都可用
    INTERVAL=2
    while true; do
        all_open=true
        for port in $(echo $ip_ports | tr ' ' '\n' | cut -d':' -f2); do
            if ! curl -s -o /dev/null -I http://$HEAD_NODE_IP:$port; then
                echo "Port $port on $HEAD_NODE_IP is not reachable."
                all_open=false
                break
            fi
        done

        if [ "$all_open" = true ]; then
            echo "All ports on $HEAD_NODE_IP are open and reachable."
            sleep $INTERVAL
            break
        else
            echo "Some ports on $HEAD_NODE_IP are not reachable. Retrying in $INTERVAL seconds..."
            sleep $INTERVAL
        fi
    done

    # 添加负载
    if [ -z "$prompt_len" ]; then
        echo "prompt_len is empty. use sharegpt dataset."
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
            --log_filename $filename_base \
            --prompt_save_path /workspace/llm-serve/Llumnix/benchmark_test/logs/prompts/sharegpt_$MODEL\_$DISTRIBUTION\_$REQ_NUM\_qps_$QPS \
            --qps $QPS
    else
        echo "Using random prompts with prompt_len: $prompt_len and response_len: $response_len"
        python -u /workspace/llm-serve/Llumnix/benchmark/benchmark_serving.py \
            --ip_ports $ip_ports \
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
            --max_request_len $max_request_len \
            --log_filename $filename_base \
            --prompt_save_path /workspace/llm-serve/Llumnix/benchmark_test/logs/prompts/gen_request_$MODEL\_$DISTRIBUTION\_$REQ_NUM\_qps_$QPS\_prompt_len_$prompt_len\_response_len_$response_len \
            --qps $QPS
    fi

    # 关闭服务
    ./kill.sh
    sleep 5
}

for sm_clock in "${sm_clocks[@]}"; do
    for mem_clock in "${mem_clocks[@]}"; do
        echo "clock : $sm_clock  $mem_clock"    # 暂时没用

        ./kill.sh
        sleep 5
        # 若PREFILL_TPS_STR和DECODE_TPS_STR均不为空，则使用pdd
        if [ "$PREFILL_TPS_STR" != "" ] && [ "$DECODE_TPS_STR" != "" ]; then
            Llumnix_benchmark_pdd
            ip_ports="$ip_ports $ip_ports_tmp"
        fi

        # 若TPS_STR不为空
        if [ "$TPS_STR" != "" ]; then
            # TPS的各个元素必须相同
            TP=${TPS[0]}
            # 判断TPS的各个元素是否相同，获取TPS的元素个数
            for tp in "${TPS[@]}"; do
                if [ "$tp" != "$TP" ]; then
                    echo "TPS的各个元素必须相同"
                    exit 1
                fi
            done

            Llumnix_benchmark ${#TPS[@]} $TP
            ip_ports="$ip_ports $ip_ports_tmp"
        fi
        
        add_workload

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