import csv
import json
import math
import os
import re
from statistics import mean
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class InstanceProfile:
    def __init__(self, instance_file_dir, limit_str='tp1', enable_pd=True, is_verbose=False):
        self.instance_file_dir = instance_file_dir
        self.enable_pd = enable_pd
        self.is_verbose = is_verbose

        self.instance_log_group = None
        self.results = {        # [inference_type][token_num] = step_time:list
            'prefill': {},
            'decode': {},
        }
        self.avg_results = {  # [inference_type][token_num] = avg_step_time
            'prefill': {},
            'decode': {},
        }

        self.fit_results = {}

        self.limit_str = limit_str

        self.cache_file = os.path.join(self.instance_file_dir, f'{limit_str}_instance_profile_results.json')
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                self.results = json.load(f)
                if is_verbose:
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
        if self.is_verbose:
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
        self.fit_token_num_vs_avg_step_time()
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
            if inference_type == 'prefill':
                coefficients = np.polyfit(token_nums, avg_step_times, 2)
            elif inference_type == 'decode':
                coefficients = np.polyfit(token_nums, avg_step_times, 2)
            self.fit_results[inference_type] = coefficients
            if self.is_verbose:
                print(f"{inference_type} fit_results:{coefficients}")
            
            # 计算拟合的均方根误差
            y_pred = np.polyval(coefficients, token_nums)
            rmse = np.sqrt(np.mean((avg_step_times - y_pred) ** 2))
            if self.is_verbose:
                print(f"RMSE: {rmse:.4f}")
    
    # 绘制token_num和avg_step_time的关系图,使用matplotlib,
    def plot_token_num_vs_avg_step_time(self):
        import matplotlib.pyplot as plt
        # 分配两个子图
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        for i, inference_type in enumerate(self.avg_results):
            token_nums = list(self.avg_results[inference_type].keys())
            avg_step_times = list(self.avg_results[inference_type].values())

            token_nums = np.array(token_nums, dtype=float)
            avg_step_times = np.array(avg_step_times, dtype=float)

            if len(token_nums) > 2:
                # 基于IQR方法剔除异常值
                q1 = np.percentile(avg_step_times, 25)
                q3 = np.percentile(avg_step_times, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")
                mask = (avg_step_times >= lower_bound) & (avg_step_times <= upper_bound)
                token_nums = token_nums[mask]
                avg_step_times = avg_step_times[mask]
            # for t1,t2 in zip(token_nums, avg_step_times):
            #     print(t1,t2)
            axs[i].scatter(token_nums, avg_step_times, label=f'{inference_type} data', color='red', s=10)
            # 拟合曲线
            if len(self.fit_results) == 0:
                self.fit_token_num_vs_avg_step_time()
            if inference_type in self.fit_results:
                fit_line = np.polyval(self.fit_results[inference_type], token_nums)
                axs[i].plot(token_nums, fit_line, label=f'{inference_type} fit', color='orange')
                # 在图片上显示拟合结果
                fit_eq = f"y = {self.fit_results[inference_type][0]:.6f}x^2 + {self.fit_results[inference_type][1]:.6f}x + {self.fit_results[inference_type][2]:.6f}"
                axs[i].text(0.05, 0.95, fit_eq, transform=axs[i].transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(facecolor='white', edgecolor='orange', alpha=0.5))
                # 在图片上显示RMSE
                y_pred = np.polyval(self.fit_results[inference_type], token_nums)
                rmse = np.sqrt(np.mean((avg_step_times - y_pred) ** 2))
                axs[i].text(0.05, 0.90, f'RMSE: {rmse:.4f}', transform=axs[i].transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(facecolor='white', edgecolor='orange', alpha=0.5))
            axs[i].set_title(f'Token Num vs Avg Step Time for {inference_type}')
            axs[i].set_xlabel('Token Num')
            axs[i].set_ylabel('Avg Step Time (ms)')
            axs[i].legend()
            axs[i].grid(True)
        plt.tight_layout()
        plt.show()
    def fit_certain_data(self, inference_type, data):
        if not isinstance(data, list):
            data = [data]
        return np.polynal(self.fit_results[inference_type], data)
    
# Decode考虑batchsize和req_len两个因素的影响
class InstanceDecodeProfiler:
    def __init__(self, instance_file_dir, limit_str='tp1', enable_pd=True, is_verbose=False):
        self.instance_file_dir = instance_file_dir
        self.enable_pd = enable_pd

        self.instance_log_group = None
        self.x_values = []  # 用于存储自变量
        self.y_values = []  # 用于存储因变量

        self.results = {}   # {(batch_size, max_seq_len): step_time:list}
        self.avg_results = {}  # {(batch_size, max_seq_len): avg_step_time}

        self.fit_results = {}

        self.limit_str = limit_str

        self.is_verbose = is_verbose

        self.cache_file = os.path.join(self.instance_file_dir, f'{limit_str}_instance_decode_profile_results.json')
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                self.results = json.load(f)
                if self.is_verbose:
                    print(f'Loaded cached results from {self.cache_file}')
                # 将keys转换为tuple类型，去掉括号
                self.results = {literal_eval(k): v for k, v in self.results.items()}
                # 查看key和value的类型
                if self.results:
                    first_key = next(iter(self.results))
                    if self.is_verbose:
                        print(f'First key type: {type(first_key)}, value type: {type(self.results[first_key])}')
        self.filter_count = 0 # 记录过滤掉的step_time数量

    def get_inference_type(self, group):
        for inference_type in group['inference_type']:
            if not pd.isna(inference_type) and inference_type != None:
                return inference_type
    
    def get_step_time(self, group, inference_type):
        assert inference_type == 'decode', "This method is only for decode inference type"
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

        # batch_sizes, max_seq_lens, seq_lens_sums = [], [], []
        # for i,row in group.iterrows():
        #     if row['seq_lens'] is None or pd.isna(row['seq_lens']):
        #         print(f"Warning: seq_lens is None or NaN for row {i}, skipping this row.")
        #         continue

        #     seq_lens = literal_eval(row['seq_lens'])
        #     inference_type = row['profiling_inference_type']

        #     batch_size = row['running_seq_lens'] if inference_type == 'prefill' else row['profiling_num_seqs']
        #     batch_size = int(batch_size)
        #     print(sum(seq_lens),batch_size,int(row['running_seq_lens']))
        #     if inference_type == 'decode' and sum(seq_lens) + batch_size != int(row['running_seq_lens']):
        #         seq_lens_pre = literal_eval(group['seq_lens'].iloc[i-1]) if i > 0 else []
        #         print(seq_lens_pre)
        #         if sum(seq_lens_pre) + batch_size == int(row['running_seq_lens']):
        #             seq_lens = seq_lens_pre
        #         else:
        #             print(f"Warning: seq_lens sum {sum(seq_lens)} + batch_size {batch_size} does not match running_seq_lens {row['running_seq_lens']} for row {i}, using previous seq_lens.")
        #     max_seq_lens.append(max(seq_lens) if seq_lens else 0)
        #     seq_lens_sums.append(sum(seq_lens) if seq_lens else 0)
        #     batch_sizes.append(batch_size)


        batch_size = group['running_seq_lens'] if inference_type == 'prefill' else group['profiling_num_seqs']
        seq_lens = group['seq_lens']
        # 将seq_lens中的元素从字符串转换为整数列表
        seq_lens = seq_lens.apply(lambda x: literal_eval(x) if pd.notnull(x) else [])
        
        seq_lens_sum = group['running_seq_lens']
        # 计算每个batch的最大seq_len
        # max_seq_lens = seq_lens.apply(lambda x: max(x) if x else 0)           # max_seq_lens
        max_seq_lens = seq_lens.apply(lambda x: sum(x) / len(x) if x else 0)    # avg_seq_len

        # 将token_nums和seq_lens合并为一个二维特征矩阵
        x_val =np.column_stack((batch_size, max_seq_lens, seq_lens_sum, seq_lens.apply(lambda x: sum(x) if x else 0)))
        y = group['last_inference_latency'].to_list()
        # 过滤掉seq_lens_sum和seq_lens差距过大的部分
        mask = x_val[:, 2] - x_val[:, 3] <= x_val[:, 0]
        x_val = x_val[mask] 
        y = np.array(y)[mask]

        self.filter_count += len(group) - len(y)
        
        assert len(x_val) == len(y), f"Length mismatch: x_val {len(x_val)} and last_inference_latency {len(y)}"
        return x_val[:, :3], y

    def profile(self, instance_file):
        if not os.path.isfile(instance_file):
            print(f"File {instance_file} does not exist.")
            return None
        if self.is_verbose:
            print(f'[profile] Processing file: {instance_file}')
        instance_log = pd.read_csv(instance_file)
        # 删除dispatch_load_metric为-inf的行
        instance_log = instance_log[instance_log['dispatch_load_metric'] != -np.inf]

        self.instance_log_group = instance_log.groupby("instance_id")
        for i, (instance_id, group) in enumerate(self.instance_log_group):
            if self.enable_pd:
                # print(f'Processing instance_id: {instance_id}, group size: {len(group)}')
                inference_type = self.get_inference_type(group)
                if inference_type != 'decode':
                    continue
                x_val, step_time = self.get_step_time(group, inference_type)
                
                for j, x in enumerate(x_val):
                    k = tuple(x)
                    if k not in self.results:
                        self.results[k] = []
                    self.results[k].append(step_time[j])

    
    def sort_results_according_to_token_num(self):
        # 对结果按照batch_size和seq_len进行排序
        sorted_items = sorted(self.results.items(), key=lambda x: (x[0][0], x[0][1], x[0][2]))
        self.results = {k: v for k, v in sorted_items if k[2] > k[0] + k[1] -1}  # 只保留seq_lens_sum大于batch_size+max_seq_len-1的项
    
    def calculate_avg_step_time(self):
        # 计算[bs, max_seq_len]的平均step_time
        for x, step_times in self.results.items():
            avg_step_time = mean(step_times)
            self.avg_results[x] = avg_step_time
        self.x_values = list(self.avg_results.keys())
        self.y_values = list(self.avg_results.values())
        self.x_values = np.array(self.x_values, dtype=float)

        self.req_len_sum = self.x_values[:, 2]  # seq_lens_sum
        self.x_values = self.x_values[:, :2]  # 只取batch_size和max_seq_len
        # self.x_values = self.x_values[:, [0,2]]  # 只取batch_size和seq_lens_sum
        self.y_values = np.array(self.y_values, dtype=float)


        

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
            # 将keys转换为字符串形式的tuple
            results_str = {str(k): v for k, v in self.results.items()}
            json.dump(results_str, f, indent=4)
        print(f'Results saved to {self.cache_file}')

    def get_pdd_instance_info(self):
        if len(self.results) == 0:
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
            self.sort_results_according_to_token_num()
            self.save_results_to_json()
        self.calculate_avg_step_time()
        self.fit_data()
        # self.print_results()
    
    # 拟合bs、max_seq_len和avg_step_time的关系
    def fit_data(self):
        if self.is_verbose:
            print(f"x_val shape: {self.x_values.shape}, y_val shape: {self.y_values.shape}")

        if len(self.x_values) < 2:
            print(f"Not enough data to fit.")
            return None
        # 剔除异常值
        if len(self.y_values) > 2:
            q1 = np.percentile(self.y_values, 25)
            q3 = np.percentile(self.y_values, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            if self.is_verbose:
                print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")
            mask = (self.y_values >= lower_bound) & (self.y_values <= upper_bound)
            self.x_values = self.x_values[mask]
            self.req_len_sum = self.req_len_sum[mask]
            self.y_values = self.y_values[mask]
        # 多元线性拟合
        model = LinearRegression()
        model.fit(self.x_values, self.y_values)
        # 拟合结果
        if self.is_verbose:
            print(f"fit_results:{model.intercept_} + {model.coef_}")
        # 计算拟合值
        y_pred = model.predict(self.x_values)
        # 计算拟合的均方根误差
        rmse = np.sqrt(np.mean((self.y_values - y_pred) ** 2))
        self.rmse = rmse
        if self.is_verbose:
            print(f"RMSE: {rmse:.4f}")
        # 保存模型
        self.model = model
        return y_pred
    
    # 绘制token_num和avg_step_time的关系图,使用matplotlib,
    def plot_result(self):
        y_pred = self.fit_data()
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.x_values[:,0], self.x_values[:,1], self.y_values, c='blue', alpha=0.3, label='Actual Data')
        # ax.scatter(self.x_values[:,0], self.x_values[:,1], y_pred, c='red', alpha=0.3, label='Fitted Data')

        # 绘制拟合平面
        # x1_grid, x2_grid = np.meshgrid(np.linspace(50, 500, 10), np.linspace(1, 16, 10))
        x1_grid, x2_grid = np.meshgrid(np.linspace(self.x_values[:,0].min(), self.x_values[:,0].max(), 10),
                                        np.linspace(self.x_values[:,1].min(), self.x_values[:,1].max(), 10))
        # 计算拟合平面的高度
        y_grid = self.model.intercept_ + self.model.coef_[0] * x1_grid + self.model.coef_[1] * x2_grid
        ax.plot_surface(x1_grid, x2_grid, y_grid, alpha=0.6, color='green')
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Max Seq Len')
        ax.set_zlabel('Avg Step Time (ms)', labelpad=0)
        ax.legend()
        plt.title('Multivariate Linear Regression Fit')
        plt.tight_layout()
        # fig.subplots_adjust(left=0.15, right=0.8, bottom=0.15, top=0.95)
        # fig.subplots_adjust(left=0.15, bottom=0.15)  # 增加边距，防止y label被遮挡
        plt.show()
        return y_pred
    
    def plot_result_single_val(self, fixed_seq_len=None, fixed_batch_size=None):
        assert fixed_seq_len is None or fixed_batch_size is None, "You can only fix one of seq_len or batch_size."
        # 固定seq_len，绘制batch_size和avg_step_time的关系图
        if len(self.x_values) == 0:
            print("No data to plot.")
            return
        if fixed_seq_len is not None:
            # 过滤出固定seq_len的数据
            mask = self.x_values[:, 1] == fixed_seq_len
            x_values_fixed = self.x_values[mask]
            y_values_fixed = self.y_values[mask]
            x_values_fixed = x_values_fixed[:,0]
            x_label = 'Batch Size'
            other_msg = f"Seq Len = {fixed_seq_len}"
        elif fixed_batch_size is not None:
            # 过滤出固定batch_size的数据
            mask = self.x_values[:, 0] == fixed_batch_size
            x_values_fixed = self.x_values[mask]
            y_values_fixed = self.y_values[mask]
            x_values_fixed = x_values_fixed[:,1]
            x_label = 'Max Seq Len'
            other_msg = f"Batch Size = {fixed_batch_size}"
        if len(x_values_fixed) == 0:
            print(f"No data found for seq_len={fixed_seq_len}.")
            return
        
        plt.scatter(x_values_fixed, y_values_fixed, color='blue', s=10, label=f'Seq Len = {fixed_seq_len}')
        plt.xlabel(x_label)
        plt.ylabel('Avg Step Time (ms)')
        plt.title(f'{x_label} vs Avg Step Time ({other_msg})')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    def fit_certain_data(self, inference_type, data):
        if not isinstance(data, list):
            data = [data]
        return np.polynal(self.fit_results[inference_type], data)

class MigrationProfile:
    def __init__(self, log_files, is_verbose=False):
        if not isinstance(log_files, list):
            log_files = [log_files]
        self.log_files = log_files
        self.results = {} # {blocks:time_ms}
        self.avg_results = {} # {blocks:avg_time_ms}
        self.fit_results = None
        self.is_verbose = is_verbose

    def extract_migration_info(self, log_file):
        if self.is_verbose:
            print(f"[extract_migration_info] Processing log file: {log_file}")

        # 检查文件是否存在
        if not os.path.isfile(log_file):
            print(f"File {log_file} does not exist.")
            return {}
        # 示例： Instance ... migrate done, migrate request ['494c45676def4572986621d1afbc337f'], migration status: MigrationStatus.FINISHED, len: 7 blocks, cost: 240.65113067626953 ms
        # 正确的正则表达式应为：
        pattern = r"migrate request \[(.*?)\].*?len: (\d+) blocks,.*?cost: ([\d\.]+) ms"
        count = 0

        # 逐行读取文件（自动处理大文件）
        with open(log_file, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                # 获取迁移时间和速度
                if 'migrate done' in line and 'cost:' in line and 'MigrationStatus.FINISHED' in line:
                    if count < 10:
                        count += 1
                        continue
                    match = re.search(pattern, line)
                    if match:
                        ids_str = match.group(1)
                        blocks = int(match.group(2))
                        time = float(match.group(3))
                        speed = blocks / time * 1000 if time > 0 else 0  # blocks/ms -> blocks/s
                        request_ids = [req_id.strip("'") for req_id in ids_str.split(', ')]
                        for req_id in request_ids:
                            if len(req_id) > 0:
                                if blocks not in self.results:
                                    self.results[blocks] = []
                                self.results[blocks].append(time)
    def get_migration_info(self):
        for log_file in self.log_files:
            self.extract_migration_info(log_file)
        sorted_items = sorted(self.results.items(), key=lambda x: x[0])
        self.results = {k: v for k, v in sorted_items}
        self.calculate_avg_migrate_time()
        self.fit_block_num_vs_avg_migration_time()

    def calculate_avg_migrate_time(self):
        for block, migration_times in self.results.items():
            avg_migration_time = mean(migration_times)
            self.avg_results[block] = avg_migration_time
    
    # 拟合block_num和avg_migration_time的关系
    def fit_block_num_vs_avg_migration_time(self):
        fit_results = {}
        block_num = list(self.avg_results.keys())
        avg_migration_time = list(self.avg_results.values())
        block_num = np.array(block_num, dtype=float)
        avg_migration_time = np.array(avg_migration_time, dtype=float)

        if len(block_num) < 2:
            print(f"Not enough data to fit.")
            return None
        # 使用numpy的polyfit进行线性拟合
        coefficients = np.polyfit(block_num, avg_migration_time, 1)
        self.fit_results = coefficients
        if self.is_verbose:
            print(f"fit_results:{self.fit_results}")
    
        # 绘制token_num和avg_step_time的关系图,使用matplotlib,
    def plot_block_num_vs_avg_migration_time(self):
        import matplotlib.pyplot as plt

        block_num = list(self.avg_results.keys())
        avg_migration_time = list(self.avg_results.values())
        block_num = np.array(block_num, dtype=float)
        avg_migration_time = np.array(avg_migration_time, dtype=float)
        # plt.scatter(block_num, avg_migration_time, color='red', s=10)
        if len(block_num) > 2:
            q1 = np.percentile(avg_migration_time, 25)
            q3 = np.percentile(avg_migration_time, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")
            mask = (avg_migration_time >= lower_bound) & (avg_migration_time <= upper_bound)
            block_num = block_num[mask]
            avg_migration_time = avg_migration_time[mask]
        plt.scatter(block_num, avg_migration_time, color='red', s=10)
        # 拟合曲线
        self.fit_block_num_vs_avg_migration_time()
        if self.fit_results is not None:
            fit_line = np.polyval(self.fit_results, block_num)
            plt.plot(block_num, fit_line, color='orange')
            # 在图片上显示拟合结果
            fit_eq = f"y = {self.fit_results[0]:.6f}x + {self.fit_results[1]:.6f}"
            plt.text(0.05, 0.95, fit_eq, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', bbox=dict(facecolor='white', edgecolor='orange', alpha=0.5))
            # 在图片上显示RMSE
            y_pred = fit_line
            rmse = np.sqrt(np.mean((avg_migration_time - y_pred) ** 2))
            plt.text(0.05, 0.88, f'RMSE: {rmse:.4f}',       transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', bbox=dict(facecolor='white', edgecolor='orange', alpha=0.5))
            # 斜率的倒数
            slope = self.fit_results[0]
            plt.text(0.05, 0.81, f'1/Slope: {1/slope:.4f}', transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', bbox=dict(facecolor='white', edgecolor='orange', alpha=0.5))
        plt.title(f'Block Num vs Avg Migration Time')
        plt.xlabel('Block Num')
        plt.ylabel('Avg Migration Time (ms)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    def fit_certain_data(self, data):
        if not isinstance(data, list):
            data = [data]
        return np.polynal(self.fit_results, data)


if __name__ == '__main__':
    # instance_file_dir = '/workspace/llm-serve/Llumnix/benchmark_test/logs/A6000-2-concurrency-1/llama-13b/poisson'
    # instance_profile = InstanceProfile(instance_file_dir, enable_pd=True)
    # instance_profile.get_pdd_instance_info()
    # instance_profile.plot_token_num_vs_avg_step_time()
    # print(instance_profile.results)

    # file = '/workspace/llm-serve/Llumnix/benchmark_test/logs/A6000-2-formal2-concurrency-1-pdd-4/llama-13b/poisson/serve_pdd_tp1_2000_qps_4_1_3.log'
    # migration_profile = MigrationProfile(file)
    # migration_profile.extract_migration_info()
    # print(migration_profile.avg_results)

    instance_file_dir = '/workspace/llm-serve/Llumnix/benchmark_test/logs/A6000-2-formal2-concurrency-1-pdd-4/llama-13b/poisson'
    instance_profile = InstanceDecodeProfiler(instance_file_dir, enable_pd=True)
    instance_profile.get_pdd_instance_info()
    # instance_profile.plot_result()
    # print(instance_profile.results)
    X = instance_profile.x_values
    y = instance_profile.y_values
    # 剔除异常值
    if len(y) > 2:
        q1 = np.percentile(y, 25)
        q3 = np.percentile(y, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")
        mask = (y >= lower_bound) & (y <= upper_bound)
        X = X[mask]
        y = y[mask]
        # 随机抽样1000个点进行可视化
        if len(X) > 2000:
            indices = np.random.choice(len(X), 2000, replace=False)
            X = X[indices]
            y = y[indices]
    # 基于plotly绘制可交互三维图
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Scatter3d(
        x=X[:, 0], y=X[:, 1], z=y,
        mode='markers',
        marker=dict(size=2, color='blue', opacity=0.1)
    )])
    # 设置图表标题和轴标签
    fig.update_layout(
        title='Batch Size and Max Seq Len vs Avg Step Time',
        scene=dict(
            xaxis_title='Batch Size',
            yaxis_title='Max Seq Len',
            zaxis_title='Avg Step Time (ms)'
        )
    )
    fig.show()