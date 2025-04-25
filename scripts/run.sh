# !/bin/bash
export HEAD_NODE_IP='127.0.0.1'

# $1: tp
# $2: 请求数量
TP=$1
REQ_NUM=$2

nvidia-smi -pl 300
nvidia-smi -pm ENABLED
nvidia-smi -acp 0
#读取MODEL，设置默认值
MODEL=${3:-"llama-2-7b"}
DISTRIBUTION=${4:-"poisson"}     # "burst", "uniform", "poisson", "gamma"
if [ "$MODEL" == "llama-2-7b" ]; then
    MODEL_PATH="/share/models/llama-2-7b"
elif [ "$MODEL" == "llama-2-13b" ]; then
    MODEL_PATH="/share/models/llama-2-13b"
elif [ "$MODEL" == "llama-7b" ]; then
    MODEL_PATH="/share/models/llama/llama-7b"
fi
echo "模型路径: $MODEL_PATH"

BASE_DIR='/workspace/llm-serve/Llumnix/logs/l40/'$MODEL/$DISTRIBUTION
mkdir -p $BASE_DIR

# 定义两个数组
# sm_clocks=("1005" "1275" "1500" "2100")    # 1005, 1275, 1500, 2100
# mem_clocks=("8001")
sm_clocks=("2490")    # "2085" "1695" "1290" "885"
# sm_clocks=("2085")    # 1005, 1275, 1500, 2100
mem_clocks=("9001")

Llumnix_benchmark() {
    # master节点
    HEAD_NODE=1 python -m llumnix.entrypoints.vllm.api_server \
                    --host 127.0.0.1 \
                    --port 1234 \
                    --initial-instances 1 \
                    --launch-ray-cluster \
                    --model $MODEL_PATH \
                    --worker-use-ray \
                    --migration-backend rayrpc \
                    --log-instance-info \
                    --tensor-parallel-size $1 \
                    --log-filename $BASE_DIR/serve_tp$1_$2_$3_$4 &

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

    # 添加负载
    python /workspace/llm-serve/Llumnix/benchmark/benchmark_serving.py \
        --ip_ports $HEAD_NODE_IP:1234 \
        --tokenizer $MODEL_PATH \
        --random_prompt_count $2 \
        --dataset_type "sharegpt" \
        --dataset_path /workspace/llm-serve/sharegpt_gpt4.jsonl \
        --distribution $DISTRIBUTION \
        --log_latencies \
        --fail_on_response_failure \
        --prompt_save_dir $BASE_DIR \
        --log_filename $BASE_DIR/benchmark_tp$1_$2_$3_$4

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