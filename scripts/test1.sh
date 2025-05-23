filename=/workspace/llm-serve/Llumnix/logs/l40-pdd--2/llama-2-7b/poisson/rayrpc/serve_pdd_tp1_20_qps_4_1_1_prompt_len_256_response_len_128.log
echo $filename
# 判断文件是否包含指定字符串
if [ -f "$filename" ] && grep -q "\[LJX\] LLMEngineLlumnix._process_request_outputs engine_step_timestamp_end" "$filename"; then
    echo "File $filename already contains the target string, skipping."
    return
fi