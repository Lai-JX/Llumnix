# filename=/workspace/llm-serve/Llumnix/benchmark_test/logs/A6000-2-formal-concurrency-2-pdd-4/llama-7b/poisson/serve_pdd_tp1_2000_qps_4_2_2_benchmark.log
# echo $filename
# # 判断文件是否包含指定字符串
# if [ -f "$filename" ] && grep -q "Error" "$filename"; then
#     echo "File $filename contains Error, skipping."
#     return
# fi

PREFILL_TPS_STR=${1:-"1,1"}
DECODE_TPS_STR=${2:-""}
# 判断DECODE_TPS_STR是否为空
if [ -z "$DECODE_TPS_STR" ]; then
    echo "DECODE_TPS_STR is empty."
fi
IFS=',' read -ra PREFILL_TPS <<< "$PREFILL_TPS_STR"
IFS=',' read -ra DECODE_TPS <<< "$DECODE_TPS_STR"
for prefill_tp in "${PREFILL_TPS[@]}"; do
    echo "Prefill TPS: $prefill_tp"
done

for decode_tp in "${DECODE_TPS[@]}"; do
    echo "Decode TPS: $decode_tp"
done