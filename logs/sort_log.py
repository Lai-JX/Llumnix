import re

def extract_timestamp(line):
    # 匹配常见的时间格式（如 2025-05-24 05:20:00,202 或 05-24 05:20:00）
    match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)|(\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
    if match:
        ts = match.group(0)
        # 统一转为可比较的字符串（补年份）
        if len(ts) == 19:  # 05-24 05:20:00
            ts = "2025-" + ts  # 假设年份为2025
        return ts
    # 匹配 float 时间戳（如 1748064018.5282085）
    match = re.search(r'\d{10}\.\d+', line)
    if match:
        return match.group(0)
    return None

def log_sort_key(line):
    ts = extract_timestamp(line)
    # print(f"Extracted timestamp: {ts}")  # Debugging line
    if ts is None:
        # print(f"No timestamp found in line: {line}")  # Debugging line
        return "2026-05-24 05:20:21,065"  # 没有时间戳的行排在最后
    # 尝试转为float（如epoch时间戳），否则按字符串排序
    try:
        return float(ts)
    except:
        return ts

def main():
    input_file = "/workspace/llm-serve/Llumnix/logs/l40-pdd--test/llama-2-13b/uniform/gloo/serve_pdd_tp2_30_qps_4_1_1_prompt_len_256_response_len_128.log"
    output_file = "/workspace/llm-serve/Llumnix/logs/l40-pdd--test/llama-2-13b/uniform/gloo/serve_pdd_tp2_30_qps_4_1_1_prompt_len_256_response_len_128_sorted.log"

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    sorted_lines = sorted(lines, key=log_sort_key)

    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(sorted_lines)

    print(f"Sorted log written to {output_file}")

if __name__ == "__main__":
    main()