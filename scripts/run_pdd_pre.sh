# !/bin/bash
export HEAD_NODE_IP='127.0.0.1'

# $1: tp
# $2: 请求数量
TP=$1
REQ_NUM=$2

BASE_DIR='/workspace/llm-serve/Llumnix/logs/l40-pdd-ratio_3'

nvidia-smi -pl 300
nvidia-smi -pm ENABLED
nvidia-smi -acp 0

# 定义两个数组
# sm_clocks=("1005" "1275" "1500" "2100")    # 1005, 1275, 1500, 2100
# mem_clocks=("8001")
sm_clocks=("2490")    # 1005, 1275, 1500, 2100
mem_clocks=("9001")
# /workspace/llm-serve/Llumnix/scripts/setup_ray.sh

Llumnix_benchmark() {
    # python -m llumnix.entrypoints.vllm.serve \
    #         --host 127.0.0.1 --port 1234 \
    #         --model /share/models/llama/llama-7b --trust-remote-code --worker-use-ray \
    #         --enable-pd-disagg --pd-ratio 3:1 --max-instances 4 \
    #         --log-instance-info --migration-backend rayrpc --enable-migration \
    #         --tensor-parallel-size $1 \
    #         --log-filename $BASE_DIR/serve_pdd_3-1_tp$1_$2_$3_$4 &
    # master节点
    HEAD_NODE=1 python -m llumnix.entrypoints.vllm.api_server \
                    --host 127.0.0.1 \
                    --port 1234 \
                    --initial-instances 1 \
                    --launch-ray-cluster \
                    --enable-pd-disagg --instance-type prefill \
                    --model /share/models/llama/llama-7b \
                    --worker-use-ray \
                    --migration-backend rayrpc \
                    --enable-migration \
                    --log-instance-info \
                    --tensor-parallel-size $1 \
                    --log-filename $BASE_DIR/serve_pdd_1_tp$1_$2_$3_$4 > output1.log 2>&1 &
    # 判断$HEAD_NODE_IP:1234是否可用 
    INTERVAL=2
    while true; do
        if curl -s -o /dev/null -I http://$HEAD_NODE_IP:1234; then
            echo "Port 1234 on $HEAD_NODE_IP is open and reachable."
            break
        else
            echo "Port 1234 on $HEAD_NODE_IP is not reachable. Retrying in $INTERVAL seconds..."
            sleep $INTERVAL
        fi
    done
    python -m llumnix.entrypoints.vllm.api_server \
                    --host 127.0.0.1 \
                    --port 1235 \
                    --initial-instances 1 \
                    --enable-pd-disagg --instance-type decode \
                    --model /share/models/llama/llama-7b \
                    --worker-use-ray \
                    --migration-backend rayrpc \
                    --enable-migration \
                    --log-instance-info \
                    --tensor-parallel-size $1 \
                    --log-filename $BASE_DIR/serve_pdd_2_tp$1_$2_$3_$4 > output2.log 2>&1 &

    python -m llumnix.entrypoints.vllm.api_server \
                    --host 127.0.0.1 \
                    --port 1236 \
                    --initial-instances 1 \
                    --enable-pd-disagg --instance-type decode \
                    --model /share/models/llama/llama-7b \
                    --worker-use-ray \
                    --migration-backend rayrpc \
                    --enable-migration \
                    --log-instance-info \
                    --tensor-parallel-size $1 \
                    --log-filename $BASE_DIR/serve_pdd_3_tp$1_$2_$3_$4 > output3.log 2>&1 &
    python -m llumnix.entrypoints.vllm.api_server \
                    --host 127.0.0.1 \
                    --port 1237 \
                    --initial-instances 1 \
                    --enable-pd-disagg --instance-type decode \
                    --model /share/models/llama/llama-7b \
                    --worker-use-ray \
                    --migration-backend rayrpc \
                    --log-instance-info \
                    --tensor-parallel-size $1 \
                    --log-filename $BASE_DIR/serve_pdd_4_tp$1_$2_$3_$4 > output4.log 2>&1 &

    # # 判断$HEAD_NODE_IP:1234是否可用 
    # INTERVAL=2
    # while true; do
    #     if curl -s -o /dev/null -I http://$HEAD_NODE_IP:1234; then
    #         echo "Port 1234 on $HEAD_NODE_IP is open and reachable."
    #         break
    #     else
    #         echo "Port 1234 on $HEAD_NODE_IP is not reachable. Retrying in $INTERVAL seconds..."
    #         sleep $INTERVAL
    #     fi
    # done
    # 定义要检查的端口数组
    PORTS_TO_CHECK=(1234 1235 1236)  # 替换为实际需要检查的端口号
    INTERVAL=2

    # 遍历每个端口进行检查
    for PORT in "${PORTS_TO_CHECK[@]}"; do
        echo "Checking if port $PORT on $HEAD_NODE_IP is available..."
        while true; do
            if curl -s -o /dev/null -I http://$HEAD_NODE_IP:$PORT; then
                echo "Port $PORT on $HEAD_NODE_IP is open and reachable."
                break
            else
                echo "Port $PORT on $HEAD_NODE_IP is not reachable. Retrying in $INTERVAL seconds..."
                sleep $INTERVAL
            fi
        done
    done

    # 添加负载
    python /workspace/llm-serve/Llumnix/benchmark/benchmark_serving.py \
        --ip_ports $HEAD_NODE_IP:1234 \
        --tokenizer /share/models/llama/llama-7b \
        --random_prompt_count $2 \
        --dataset_type "sharegpt" \
        --dataset_path /workspace/llm-serve/sharegpt_gpt4.jsonl \
        --distribution "poisson" \
        --log_latencies \
        --fail_on_response_failure \
        --max_request_len 2048 \
        --log_filename $BASE_DIR/benchmark_pdd_tp$1_$2_$3_$4

    # 关闭服务
    ./kill.sh
    sleep 5
}

# 外层循环遍历第一个数组
for sm_clock in "${sm_clocks[@]}"; do
    # 内层循环遍历第二个数组
    for mem_clock in "${mem_clocks[@]}"; do
        echo "clock : $sm_clock  $mem_clock"

        # 生成从0到n-1的数字序列
        device_ids=""
        for ((i=0; i < $TP; i++)); do
            device_ids="$device_ids,$i"
        done
        # 去掉开头的逗号
        device_ids="${device_ids:1}"
        # 设置频率
        nvidia-smi -i $device_ids -lgc $sm_clock
        nvidia-smi -i $device_ids -lmc $mem_clock
        nvidia-smi -i $device_ids -ac $mem_clock,$sm_clock

        # 运行基准测试
        Llumnix_benchmark $TP $REQ_NUM $sm_clock $mem_clock

        # 恢复设置
        nvidia-smi -i $device_ids -acp 1
        nvidia-smi -i $device_ids -rac 
        nvidia-smi -i $device_ids -rgc 
        nvidia-smi -i $device_ids -rmc
    done
done

# 恢复设置
nvidia-smi -i $device_ids -acp 1
nvidia-smi -i $device_ids -rac 
nvidia-smi -i $device_ids -rgc 
nvidia-smi -i $device_ids -rmc