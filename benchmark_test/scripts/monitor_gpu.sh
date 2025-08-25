#!/bin/bash

# 配置参数
CHECK_INTERVAL=10          # 检测间隔（秒），建议10秒以减少开销
THRESHOLD=10               # 利用率阈值（%）
REQUIRED_CONSECUTIVE=60    # 需要连续满足条件的次数（10分钟=60次*10秒）

# 初始化计数器
consecutive_low=0
total_checks=0

first_time=true

# 重启进程函数
restart_process() {
    ./kill_all.sh
    echo "Restart..."
    nohup ./run_qps.sh > run_tmp.log 2>&1 &  # 后台运行run.sh[5](@ref)[10](@ref)
    echo "Restarted"
}

echo "Starting GPU monitoring: Checking if utilization stays below ${THRESHOLD}% for ${REQUIRED_CONSECUTIVE} consecutive checks (≈10 minutes)..."

while true; do
    # 获取所有GPU的利用率（兼容多GPU场景）
    utilizations=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    all_below_threshold=true
    count=0
    # 检查每个GPU的利用率
    for util in $utilizations; do
        if [ "$util" -ge "$THRESHOLD" ]; then
            count=$((count + 1))
            # 有4个GPU利用率高于阈值，标记为false
            if [ "$count" -ge 4 ]; then
                all_below_threshold=false
                break
            fi
        fi
    done

    # 更新连续计数
    timestamp=$(date +'%Y-%m-%d %H:%M:%S')
    if [ "$all_below_threshold" = true ]; then
        consecutive_low=$((consecutive_low + 1))
        echo "[${timestamp}] Check ${total_checks} more than 4 GPU(s) below threshold - Consecutive count: ${consecutive_low}"
    else
        consecutive_low=0
        echo "[${timestamp}] Check ${total_checks} 4 GPU(s) above threshold - Reset counter"
    fi

    # 第一次检测时启动进程
    if [ "$first_time" = true ]; then
        first_time=false
        echo "[${timestamp}] First time check - Starting process"
        ./kill_all.sh
        sleep 2
        restart_process
    fi

    # 判断是否满足连续条件
    if [ "$consecutive_low" -ge "$REQUIRED_CONSECUTIVE" ]; then
        echo "[ALERT] All GPUs have been below ${THRESHOLD}% utilization for ${REQUIRED_CONSECUTIVE} consecutive checks (≈10 minutes)."
        # 此处可添加后续操作，如发送通知或执行任务
        ./kill_all.sh
        sleep 2
        restart_process
        consecutive_low=0  # 重置计数器以避免重复触发
    fi
    total_checks=$((total_checks + 1))
    sleep $CHECK_INTERVAL
done
