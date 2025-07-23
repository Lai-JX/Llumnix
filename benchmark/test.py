import os
from transformers import AutoTokenizer
def get_max_request_len(tokenizer, max_request_len):
    # 直接通过 tokenizer 获取
    tokenizer_tmp = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

    max_request_len = tokenizer_tmp.model_max_length
    print(f'max_request_len1:{max_request_len}')
    # 如果 model_max_length 无效，则通过配置获取
    if max_request_len > 1e6:  # 检查是否异常大
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(tokenizer)
        print(config)
        # 尝试不同参数名
        max_request_len = getattr(config, "max_position_embeddings", None)
        print(f'max_request_len2:{max_request_len}')
        if max_request_len is None:
            max_request_len = getattr(config, "n_positions", max_request_len)  # 默认值可选
    return max_request_len

if __name__ == '__main__':
    # max_request_len = get_max_request_len('/share/models/llama/llama-13b', 8192)
    path = '/workspace/llm-serve/Llumnix/benchmark_test/logs/A6000-2-formal2-concurrency-2-pdd-2/llama-30b/poisson/serve_pdd_tp2_2000_qps_1_1_1'
    file_name = os.path.splitext(path)[0] + "_latency_info.json"
    print(file_name)