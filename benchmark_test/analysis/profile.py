import csv
import json
import math
import os
import re
from statistics import mean
import pandas as pd
import numpy as np
from ast import literal_eval

class InstanceProfile:
    def __init__(self, instance_file_dir, limit_str='tp1', enable_pd=True):
        self.instance_file_dir = instance_file_dir
        self.enable_pd = enable_pd

        self.instance_log_group = None
        self.results = {        # [inference_type][token_num] = step_time:list
            'prefill': {},
            'decode': {},
        }
        self.avg_results = {  # [inference_type][token_num] = avg_step_time
            'prefill': {},
            'decode': {},
        }

        self.limit_str = limit_str

        self.cache_file = os.path.join(self.instance_file_dir, f'{limit_str}_instance_profile_results.json')
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                self.results = json.load(f)
                print(f'Loaded cached results from {self.cache_file}')
                # 将keys转换为float类型
                for inference_type in self.results:
                    self.results[inference_type] = {float(k): v for k, v in self.results[inference_type].items()}

    def get_inference_type(self, group):
        for inference_type in group['inference_type']:
            if not pd.isna(inference_type) and inference_type != None:
                return inference_type
    
    def get_step_time(self, group, inference_type):
        group = group.copy()
        # 将profiling_data列(inference_type,num_seqs,running_seq_lens,last_inference_latency)中的内容转化为4列
        group[['profiling_inference_type', 'profiling_num_seqs', 'running_seq_lens', 'last_inference_latency']] = (
            group['profiling_data']
            .apply(lambda x: literal_eval(x) if pd.notnull(x) else ("", None, None, None))
            .apply(pd.Series)
        )
        # 剔除last_inference_latency为NaN或0的行
        group = group[group['last_inference_latency'].notna() & (group['last_inference_latency'] > 0)]
        # 过滤出指定inference_type的数据
        group = group[group['profiling_inference_type'] == inference_type]
        # 根据last_inference_latency去重
        group = group.drop_duplicates(subset=['last_inference_latency'])
        token_num = group['running_seq_lens'] if inference_type == 'prefill' else group['profiling_num_seqs']
        return token_num.to_list(), group['last_inference_latency'].to_list()

    def profile(self, instance_file):
        if not os.path.isfile(instance_file):
            print(f"File {instance_file} does not exist.")
            return None
        print(f'[profile] Processing file: {instance_file}')
        instance_log = pd.read_csv(instance_file)
        # 删除dispatch_load_metric为-inf的行
        instance_log = instance_log[instance_log['dispatch_load_metric'] != -np.inf]

        self.instance_log_group = instance_log.groupby("instance_id")
        for i, (instance_id, group) in enumerate(self.instance_log_group):
            if self.enable_pd:
                inference_type = self.get_inference_type(group)
                token_num, step_time = self.get_step_time(group, inference_type)
                for j in range(len(token_num)):
                    if token_num[j] not in self.results[inference_type]:
                        self.results[inference_type][token_num[j]] = []
                    self.results[inference_type][token_num[j]].append(step_time[j])

    
    def sort_resulte_according_to_token_num(self):
        # 对结果按照token_num进行排序
        for inference_type in self.results:
            sorted_items = sorted(self.results[inference_type].items(), key=lambda x: x[0])
            self.results[inference_type] = {k: v for k, v in sorted_items}
    
    def calculate_avg_step_time(self):
        # 计算每个inference_type和token_num的平均step_time
        for inference_type in self.results:
            for token_num, step_times in self.results[inference_type].items():
                avg_step_time = mean(step_times)
                self.avg_results[inference_type][token_num] = avg_step_time

    def print_results(self, num=100):
        # 打印结果
        for inference_type in self.avg_results:
            print(f'Inference Type: {inference_type}')
            count = 0
            for token_num, avg_step_time in self.avg_results[inference_type].items():
                if count >= num:
                    break
                count += 1
                print(f'Token Num: {token_num}, Avg Step Time: {avg_step_time:.4f} ms')
            print('-' * 50)
    
    def save_results_to_json(self):
        # 将结果保存到JSON文件
        with open(self.cache_file, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f'Results saved to {self.cache_file}')

    def get_pdd_instance_info(self):
        if len(self.results['decode']) == 0:
            # 遍历 self.instance_file_dir 下所有包含pdd的csv文件
            if self.enable_pd:
                instance_files = [f for f in os.listdir(self.instance_file_dir) if self.limit_str in f and 'benchmark' not in f and f.endswith('.csv') and 'pdd' in f ]
            else:
                instance_files = [f for f in os.listdir(self.instance_file_dir) if self.limit_str in f and 'benchmark' not in f and f.endswith('.csv')]
            if not instance_files:
                print(f"No instance files found in {self.instance_file_dir} containing 'pdd'.")
                return None 
            for instance_file in instance_files:
                instance_file_path = os.path.join(self.instance_file_dir, instance_file)
                self.profile(instance_file_path)
            self.sort_resulte_according_to_token_num()
            self.save_results_to_json()
        self.calculate_avg_step_time()
        # self.print_results()
    
    # 拟合token_num和avg_step_time的关系
    def fit_token_num_vs_avg_step_time(self):
        fit_results = {}
        for inference_type in self.avg_results:
            token_nums = list(self.avg_results[inference_type].keys())
            avg_step_times = list(self.avg_results[inference_type].values())
            token_nums = np.array(token_nums, dtype=float)
            avg_step_times = np.array(avg_step_times, dtype=float)

            if len(token_nums) < 2:
                print(f"Not enough data to fit for {inference_type}.")
                continue
            # 使用numpy的polyfit进行线性拟合
            coefficients = np.polyfit(token_nums, avg_step_times, 3)
            fit_results[inference_type] = coefficients
        return fit_results
    
    # 绘制token_num和avg_step_time的关系图,使用matplotlib,
    def plot_token_num_vs_avg_step_time(self):
        import matplotlib.pyplot as plt
        # 分配两个子图
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        for i, inference_type in enumerate(self.avg_results):
            token_nums = list(self.avg_results[inference_type].keys())
            avg_step_times = list(self.avg_results[inference_type].values())
            axs[i].scatter(token_nums, avg_step_times, label=f'{inference_type} data', color='red', s=10)
            # 拟合曲线
            fit_results = self.fit_token_num_vs_avg_step_time()
            if inference_type in fit_results:
                fit_line = np.polyval(fit_results[inference_type], token_nums)
                axs[i].plot(token_nums, fit_line, label=f'{inference_type} fit', color='orange')
            axs[i].set_title(f'Token Num vs Avg Step Time for {inference_type}')
            axs[i].set_xlabel('Token Num')
            axs[i].set_ylabel('Avg Step Time (ms)')
            axs[i].legend()
            axs[i].grid(True)
        plt.tight_layout()
        plt.show()
    


if __name__ == '__main__':
    instance_file_dir = '/workspace/llm-serve/Llumnix/benchmark_test/logs/A6000-2-concurrency-1/llama-13b/poisson'
    instance_profile = InstanceProfile(instance_file_dir, enable_pd=True)
    instance_profile.get_pdd_instance_info()
    instance_profile.plot_token_num_vs_avg_step_time()
    # print(instance_profile.results)