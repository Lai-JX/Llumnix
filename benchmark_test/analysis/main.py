import csv
import json
import math
import os
import re
from statistics import mean
import pandas as pd
import numpy as np
from ast import literal_eval

class InstanceMetricsAnalysis:
    def __init__(self, instance_file, enable_pd=True):
        self.instance_file = instance_file
        self.enable_pd = enable_pd

        self.instance_log_group = None
        self.results = None

    def get_inference_type(self, group):
        for inference_type in group['inference_type']:
            if not pd.isna(inference_type) and inference_type != None:
                return inference_type
    def get_gpu_msg(self, group, metric='sm_active'):
        group = group[group['sm_active'] != "[]"].copy()
        group['sm_active'] = group['sm_active'].apply(
                        lambda x: x[0] if isinstance(x, list) and len(x) == 1 else (mean(literal_eval(x)) if isinstance(x, str) and x.startswith('[') else x)
                    )
        # print(group['sm_active'])
        return round(mean(group['sm_active']),6)

    # mofc(memory occupy for compute)) 所有块中被计算占用的比例
    def get_mofc(self, group, filter_prefill=False):
        group = group.copy()
        if filter_prefill:
            group = group[group['inference_type'] == 'prefill']
        group['running_seq_lens'] = group['running_seq_lens'].astype(float)
        group['num_available_gpu_blocks'] = group['num_available_gpu_blocks'].astype(int)
        max_blocks = group['num_available_gpu_blocks'].max()
        # 计算分母
        denominator = max_blocks - group['num_available_gpu_blocks']
        # 防止除零
        denominator = denominator.replace(0, np.nan)
        group['mofc'] = np.where(
            denominator > 0,
            np.ceil(group['running_seq_lens'] / 16) / denominator,
            1.0
        )
        group['mofc'] = group['mofc'].fillna(1.0)
        return round(group['mofc'].mean(), 6)
    
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
        mofc = self.get_mofc(group)
        # 过滤出指定inference_type的数据
        group = group[group['profiling_inference_type'] == inference_type]
        # 根据last_inference_latency去重
        group = group.drop_duplicates(subset=['last_inference_latency'])
        return {
            "mofc":mofc, 
            f"{inference_type}_step_time": round(mean(group['last_inference_latency']), 6) if not group.empty else 0.0
        }

    def get_instance_metrics(self):
        if not os.path.isfile(self.instance_file):
            print(f"File {self.instance_file} does not exist.")
            return None
        print(f'[get_instance_metrics] Processing file: {self.instance_file}')
        instance_log = pd.read_csv(self.instance_file)
        # 删除dispatch_load_metric为-inf的行
        instance_log = instance_log[instance_log['dispatch_load_metric'] != -np.inf]

        self.instance_log_group = instance_log.groupby("instance_id")
        self.results = {}
        for i, (instance_id, group) in enumerate(self.instance_log_group):
            if self.enable_pd:
                inference_type = self.get_inference_type(group)
                new_data = group[group['inference_type'] == inference_type]
                res = {
                    'inference_type': inference_type,
                    f'{inference_type}_bs': round(mean(new_data['bs']),4),
                    f'{inference_type}_all_time_bs': round(mean(group['bs']),4),
                    'gpu_cache_usage': round(mean(group['gpu_cache_usage']),6),
                    'num_running_requests': round(mean(group['num_running_requests']),6),
                    'num_waiting_requests': round(mean(group['num_waiting_requests']),6),
                    'num_killed_requests': round(mean(group['num_killed_requests']),6),
                    'sm_active' : self.get_gpu_msg(group),
                    # 'mofc': self.get_mofc(group),
                    # f'{inference_type}_step_time': self.get_step_time(group, inference_type),
                }
                res = {**res, **self.get_step_time(group, inference_type)}
            else:
                decode_data = group[group['inference_type'] == 'decode']
                res = {
                    'decode_bs': round(mean(decode_data['bs']),4),
                    'decode_ratio': round(len(decode_data) / len(group),4),
                    'gpu_cache_usage': round(mean(group['gpu_cache_usage']),6),
                    'num_running_requests': round(mean(group['num_running_requests']),6),
                    'num_waiting_requests': round(mean(group['num_waiting_requests']),6),
                    'num_killed_requests': round(mean(group['num_killed_requests']),6),
                    'sm_active' : self.get_gpu_msg(group),
                    # 'mofc': self.get_mofc(group),
                    # 'prefill_mofc': self.get_mofc(group, filter_prefill=True),
                    # f'prefill_step_time': self.get_step_time(group, 'prefill'),
                    # f'decode_step_time': self.get_step_time(group, 'decode'),
                }
                res = {**res, **self.get_step_time(group, 'prefill'), **self.get_step_time(group, 'decode')}
            self.results[instance_id] = res




class LogAnalysis:
    def __init__(self, model):
        self.path_tmp = 'A6000-2-formal2'
        self.model = model
        self.distribution = 'poisson'
        self.instance_num = 4
        self.concurrencies = [1,2,4,8,16]
        if self.model == 'llama-7b':
            self.qps = [2,4,6,8,10,12]
        elif self.model == 'llama-13b':
            self.qps = [1,2,4,]
        json_files = self.get_json_file_path(self.concurrencies[0], self.qps[0], 1, 4, ['1,1-2'])
        self.labels = json_files.keys()

        self.cache_file = f'results/latency_results_cache-{self.path_tmp}-{model}-{self.concurrencies}-{self.qps}.json'
        if os.path.exists(self.cache_file):
            print(f'[LogAnalysis] exist cache_file:{self.cache_file}')
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                self.results = json.load(f)
        else:
            self.results = {}
        
    def save_to_cache_file(self):
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

    def get_label(self, prefill_tps, decode_tps, is_pd=True):
        if not is_pd:
            res = ",".join(str(x) for x in prefill_tps)
        else:
            res = ",".join(str(x) for x in prefill_tps) + '-' + ",".join(str(x) for x in decode_tps)
        return res
    
    def get_json_file_path(self, concurrency, qps, tp, instance_num, exist_labels=None):
        file_paths = {}

        # 非pd分离的json path
        json_file = f'/workspace/llm-serve/Llumnix/benchmark_test/logs/{self.path_tmp}-concurrency-{concurrency}-pdd-{instance_num}/{self.model}/{self.distribution}/benchmark_{instance_num}_tp{tp}_2000_qps_{qps}_latency_info.json'
        file_paths[self.get_label([tp]*instance_num, None, False)] = json_file

        # pd分离的json path
        for prefill_num in range(1,instance_num):
            decode_num = instance_num-prefill_num
            json_file = f'/workspace/llm-serve/Llumnix/benchmark_test/logs/{self.path_tmp}-concurrency-{concurrency}-pdd-{instance_num}/{self.model}/{self.distribution}/benchmark_pdd_tp{tp}_2000_qps_{qps}_{prefill_num}_{decode_num}_latency_info.json'
            file_paths[self.get_label([tp]*prefill_num, [tp]*decode_num)] = json_file

        # prefill和decode tp数不同的json path
        if exist_labels is not None:
            for label in exist_labels:

                json_file = f'/workspace/llm-serve/Llumnix/benchmark_test/logs/{self.path_tmp}-pdd-hetero-concurrency-{concurrency}/{self.model}/poisson/benchmark_pdd_2000_qps_{qps}_{label.replace("-","_")}_latency_info.json'
                file_paths[label] = json_file
        return file_paths

    def get_log_file_path(self, concurrency, qps, tp, instance_num, exist_labels=None):
        file_paths = {}

        # 非pd分离的log path
        log_file = f'/workspace/llm-serve/Llumnix/benchmark_test/logs/{self.path_tmp}-concurrency-{concurrency}-pdd-{instance_num}/{self.model}/{self.distribution}/serve_{instance_num}_tp{tp}_2000_qps_{qps}.log'
        file_paths[self.get_label([tp]*instance_num, None, False)] = log_file

        # pd分离的log path
        for prefill_num in range(1,instance_num):
            decode_num = instance_num-prefill_num
            log_file = f'/workspace/llm-serve/Llumnix/benchmark_test/logs/{self.path_tmp}-concurrency-{concurrency}-pdd-4/{self.model}/poisson/serve_pdd_tp{tp}_2000_qps_{qps}_{prefill_num}_{decode_num}.log'
            file_paths[self.get_label([tp]*prefill_num, [tp]*decode_num)] = log_file

        # prefill和decode tp数不同的log path
        if exist_labels is not None:
            for label in exist_labels:
                log_file = f'/workspace/llm-serve/Llumnix/benchmark_test/logs/{self.path_tmp}-pdd-hetero-concurrency-{concurrency}/{self.model}/poisson/serve_pdd_2000_qps_{qps}_{label.replace("-","_")}.log'
                file_paths[label] = log_file
        return file_paths
    
    def get_instance_file_path(self, concurrency, qps, tp, instance_num, exist_labels=None):
        file_paths = {}

        # 非pd分离的instance path
        instance_file = f'/workspace/llm-serve/Llumnix/benchmark_test/logs/{self.path_tmp}-concurrency-{concurrency}-pdd-{instance_num}/{self.model}/{self.distribution}/serve_{instance_num}_tp{tp}_2000_qps_{qps}_instance.csv'
        file_paths[self.get_label([tp]*instance_num, None, False)] = instance_file

        # pd分离的instance path
        for prefill_num in range(1,instance_num):
            decode_num = instance_num-prefill_num
            instance_file = f'/workspace/llm-serve/Llumnix/benchmark_test/logs/{self.path_tmp}-concurrency-{concurrency}-pdd-4/{self.model}/poisson/serve_pdd_tp{tp}_2000_qps_{qps}_{prefill_num}_{decode_num}_instance.csv'
            file_paths[self.get_label([tp]*prefill_num, [tp]*decode_num)] = instance_file

        # prefill和decode tp数不同的instance path
        if exist_labels is not None:
            for label in exist_labels:
                instance_file = f'/workspace/llm-serve/Llumnix/benchmark_test/logs/{self.path_tmp}-pdd-hetero-concurrency-{concurrency}/{self.model}/poisson/serve_pdd_2000_qps_{qps}_{label.replace("-","_")}_instance.csv'
                file_paths[label] = instance_file
        return file_paths

    def get_lantency(self, json_file):
        '''
            return request_time, prefill_time, decode_time
        '''
        if not os.path.isfile(json_file):
            print(f"File {json_file} does not exist.")
            return None
        print(f'[get_lantency] Processing file: {json_file}')
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f'error:{str(e)}')
        assert len(data) == 1, "Expected data to contain only one entry"
        latencies = data[0]
        req_latencies, prefill_latencies, decode_latencies = latencies['request_latencies'], latencies['prefill_token_latencies'], latencies['decode_token_latencies']
        
        per_token_latency_breakdown_list = data[0]['per_token_latency_breakdown_list']
        prefill_waiting_time = [(per_token_latency_breakdown_list[i][0]['engine_step_timestamp_begin'] - per_token_latency_breakdown_list[i][0]['engine_add_request_timestamp'])*1000
                                for i in range(len(per_token_latency_breakdown_list))]
        engine_step_latency_prefill = [per_token_latency_breakdown_list[i][0]['engine_step_latency'] for i in range(len(per_token_latency_breakdown_list))]
        engine_step_latency_decode = [
            mean([token['engine_step_latency'] for token in per_token_latency_breakdown_list[i][1:]])
            if len(per_token_latency_breakdown_list[i][1:]) > 0 else None
            for i in range(len(per_token_latency_breakdown_list))
        ]

        return {
            'request_time':round(mean(req_latencies), 4), 
            'prefill_time': round(mean(prefill_latencies), 4),
            'decode_time': round(mean(decode_latencies), 4),
            'prefill_waiting_time': round(mean(prefill_waiting_time), 4), 
            'prefill_step_time': round(mean(engine_step_latency_prefill), 4), 
            'decode_step_time': round(mean([x for x in engine_step_latency_decode if x is not None]), 4),
        }
    
    def get_step_lantency(self, json_file):
        '''
            return prefill_waiting_time, prefill_step_time, decode_step_time
        '''
        if not os.path.isfile(json_file):
            print(f"File {json_file} does not exist.")
            return None
        print(f'[get_step_lantency] Processing file: {json_file}')
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f'error:{str(e)}')
        assert len(data) == 1, "Expected data to contain only one entry"
        per_token_latency_breakdown_list = data[0]['per_token_latency_breakdown_list']
        prefill_waiting_time = [(per_token_latency_breakdown_list[i][0]['engine_step_timestamp_begin'] - per_token_latency_breakdown_list[i][0]['engine_add_request_timestamp'])*1000
                                for i in range(len(per_token_latency_breakdown_list))]
        engine_step_latency_prefill = [per_token_latency_breakdown_list[i][0]['engine_step_latency'] for i in range(len(per_token_latency_breakdown_list))]
        engine_step_latency_decode = [
            mean([token['engine_step_latency'] for token in per_token_latency_breakdown_list[i][1:]])
            if len(per_token_latency_breakdown_list[i][1:]) > 0 else None
            for i in range(len(per_token_latency_breakdown_list))
        ]

        return {
            'prefill_waiting_time': round(mean(prefill_waiting_time), 2), 
            'prefill_step_time': round(mean(engine_step_latency_prefill), 2), 
            'decode_step_time': round(mean([x for x in engine_step_latency_decode if x is not None]), 2)
        }
    

    def extract_migration_info(self, path, verbose=False):
        '''
        tp_hetero:不用
        res = {
            'avg_speed': avg_speed,
            'avg_migration_time': avg_migration_time,
            'avg_migrate_waiting_time': avg_migrate_waiting_time,
            'avg_migration_count': avg_migration_count,
            'avg_migration_aborted_dst_count': avg_migration_aborted_dst_count,
            'sum_migration_aborted_dst_count': avg_migration_aborted_dst_count*len(migration_info),
            'reject_migrate_out_count': reject_migrate_out_count,
            'reject_migrate_in_count': reject_migrate_in_count,
        }
        '''
        print(f"[extract_migration_info] Processing log file: {path}")

        # 检查文件是否存在
        if not os.path.isfile(path):
            print(f"File {path} does not exist.")
            return {}

        migration_info = {}
        reject_migrate_in_count = 0
        reject_migrate_out_count = 0
        # 示例： Instance ... migrate done, migrate request ['494c45676def4572986621d1afbc337f'], migration status: MigrationStatus.FINISHED, len: 7 blocks, cost: 240.65113067626953 ms
        # 正确的正则表达式应为：
        pattern = r"migrate request \[(.*?)\].*?len: (\d+) blocks,.*?cost: ([\d\.]+) ms"
        count = 0

        # 逐行读取文件（自动处理大文件）
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()

                if 'reject new migrate out' in line:
                    reject_migrate_out_count += 1
                if 'reject new migrate in' in line:
                    reject_migrate_in_count += 1
                # 获取迁移时间和速度
                if 'migrate done' in line and 'cost:' in line:
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
                                assert req_id in migration_info, f"{req_id},{type(req_id)},{line}"
                                migration_info[req_id]["blocks"] = blocks
                                migration_info[req_id]["time_ms"] = time
                                migration_info[req_id]["speed_blocks_per_s"] = speed

                if "engine_step_timestamp_end" in line or "_migrate_out_one_request start" in line \
                    or "MigrationStatus.ABORTED_DST, timestamps" in line \
                        or "MigrationStatus.ABORTED_SRC, timestamps" in line :
                    # 使用正则表达式提取请求 ID
                    request_id_match = re.search(r'[0-9a-f]{32}', line)
                    # 使用正则表达式提取时间戳
                    timestamp_match = re.search(r'timestamps: \d+\.\d+', line)

                    if request_id_match and timestamp_match:
                        request_id = request_id_match.group()
                        timestamp = float(timestamp_match.group().split(":")[1])

                        # 如果请求 ID 不在字典中，则初始化一个条目
                        if request_id not in migration_info:
                            migration_info[request_id] = {
                                "blocks": 0,
                                "time_ms":  0.0,
                                "speed_blocks_per_s": 0.0,
                                "engine_step_timestamp_end": None,
                                "migrate_out_one_request_start": None,
                                "migrate_start_count": 0,
                                "ABORTED_DST_count":0,
                                "ABORTED_SRC_count":0,
                            }

                        # 根据日志行内容更新对应的时间戳
                        if "engine_step_timestamp_end" in line:
                            migration_info[request_id]["engine_step_timestamp_end"] = timestamp
                        if migration_info[request_id]["time_ms"] == 0.0:
                            if "_migrate_out_one_request start" in line:
                                migration_info[request_id]["migrate_out_one_request_start"] = timestamp
                                migration_info[request_id]["migrate_start_count"] += 1
                            elif "MigrationStatus.ABORTED_DST, timestamps" in line:
                                migration_info[request_id]["ABORTED_DST_count"] += 1
                            elif "MigrationStatus.ABORTED_SRC, timestamps" in line:
                                migration_info[request_id]["ABORTED_SRC_count"] += 1
                                
                            if migration_info[request_id]["migrate_out_one_request_start"] is not None and migration_info[request_id]["engine_step_timestamp_end"] is not None:
                                migration_info[request_id]["migrate_waiting_time"] = (migration_info[request_id]["migrate_out_one_request_start"] - migration_info[request_id]["engine_step_timestamp_end"]) *1000
                                migration_info[request_id]["migrate_waiting_time"] = max(0, migration_info[request_id]["migrate_waiting_time"])
                                # migrate_waiting_times.append(migration_info[request_id]["migrate_waiting_time"])
                        else:
                            # print("not first migration")
                            pass

        fail_req_id = set()
        for req_id, info in migration_info.items():
            if 'migrate_waiting_time' not in migration_info[req_id]:
                fail_req_id.add(req_id)
                if verbose:
                    print(f'fail req_id:{req_id}, no migrate_waiting_time, {migration_info[req_id]}')
            else:
                if migration_info[req_id]["migrate_waiting_time"] > 1000:
                    pass
                    # file_output.write(f'req_id:{req_id},migrate_waiting_time:{migration_info[req_id]["migrate_waiting_time"]},ABORTED_DST_count:{migration_info[req_id]["ABORTED_DST_count"]}')
            if migration_info[req_id]['time_ms'] == 0.0:
                fail_req_id.add(req_id)
                if verbose:
                    print(f'fail req_id:{req_id}, no migrate_time, {migration_info[req_id]}')
        for req_id in fail_req_id:
            del migration_info[req_id]
        
        if migration_info:
            avg_speed = mean(info['speed_blocks_per_s'] for info in migration_info.values())
            avg_migration_time = mean(info['time_ms'] for info in migration_info.values())
            avg_migrate_waiting_time = mean(info["migrate_waiting_time"] for info in migration_info.values())
            avg_migration_count = mean(info['migrate_start_count'] for info in migration_info.values())
            avg_migration_aborted_dst_count = mean(info['ABORTED_DST_count'] for info in migration_info.values())
            if verbose:
                print(f"fail req num(lose msg): {len(fail_req_id)}, finished_flag:{finished_flag},finished_str:{finished_str}")
                print(f"Average migration speed: {avg_speed:.2f} blocks/s")
                print(f"Average migration time: {avg_migration_time:.2f} ms")
                print(f'Average migration waiting time: {avg_migrate_waiting_time:.2f} ms')
                print(f"Average migration count: {avg_migration_count}")
                print(f"Average migration ABORTED_DST count: {avg_migration_aborted_dst_count}")
                print(f"Sum migration ABORTED_DST count: {avg_migration_aborted_dst_count*len(migration_info)}")
                print(f"reject_migrate_out_count:{reject_migrate_out_count}")
                print(f"reject_migrate_in_count:{reject_migrate_in_count}")
                print(f"max block num : {max(info['blocks'] for info in migration_info.values())}")
            # if avg_migration_count > avg_migration_aborted_dst_count + 
            for req_id, info in migration_info.items():
                info['avg_speed_blocks_per_s'] = avg_speed
        else:
            print("No migration information found.")

        # assert finished_flag
        res = {
            'avg_speed': round(avg_speed,4),
            'avg_migration_time': round(avg_migration_time,4),
            'avg_migrate_waiting_time': round(avg_migrate_waiting_time,4),
            'avg_migration_count': round(avg_migration_count,4),
            'avg_migration_aborted_dst_count': round(avg_migration_aborted_dst_count,4),
            'sum_migration_aborted_dst_count': round(avg_migration_aborted_dst_count*len(migration_info),4),
            'reject_migrate_out_count': round(reject_migrate_out_count,4),
            'reject_migrate_in_count': round(reject_migrate_in_count,4),
        }
        return res
    
    def get_all_msg_qps(self, concurrency, qps):
        print(f"[get_all_msg_qps] concurrency:{concurrency}, qps:{qps}")
        json_files = self.get_json_file_path(concurrency, qps, 1, 4, ['1,1-2'])
        log_files = self.get_log_file_path(concurrency, qps, 1, 4, ['1,1-2'])
        instance_files = self.get_instance_file_path(concurrency, qps, 1, 4, ['1,1-2'])
        labels = json_files.keys()
        res = {}
        for label in labels:
            is_pd = '-' in label
            latency = self.get_lantency(json_files[label])
            instance_ana = InstanceMetricsAnalysis(instance_files[label], is_pd)
            instance_ana.get_instance_metrics()
            if is_pd:
                migration_info = self.extract_migration_info(log_files[label],)
                res[label] = {**latency, **migration_info, **instance_ana.results}
            else:
                res[label] = {**latency, **instance_ana.results}
            print()
        return res
 
    def get_all_msg(self):
        for concurrency in self.concurrencies:
            concurrency = str(concurrency)
            if concurrency not in self.results:
                self.results[concurrency] = {}
            for q in self.qps:
                q = str(q)
                if concurrency in self.results and q in self.results[concurrency]:
                    continue
                results_qps = self.get_all_msg_qps(concurrency, q)
                self.results[concurrency][q] = results_qps
                self.save_to_cache_file()

    def get_all_msg_qps_updata_instance_metric(self, concurrency, qps, data):
        print(f"[get_all_msg_qps_updata_instance_metric] concurrency:{concurrency}, qps:{qps}")
        instance_files = self.get_instance_file_path(concurrency, qps, 1, 4, ['1,1-2'])
        labels = instance_files.keys()
        res = {}
        for label in labels:
            is_pd = '-' in label
            instance_ana = InstanceMetricsAnalysis(instance_files[label], is_pd)
            instance_ana.get_instance_metrics()
            data[label].update(instance_ana.results)
            print()
        return res
    
    def get_all_msg_updata_instance_metric(self):
        for concurrency in self.concurrencies:
            for q in self.qps:
                if str(concurrency) in self.results and str(q) in self.results[str(concurrency)]:
                    self.get_all_msg_qps_updata_instance_metric(concurrency, q, self.results[str(concurrency)][str(q)])
                    self.save_to_cache_file()

    def get_instance_split_metric_base(self, data, metric):

        metric_split = metric.split('-')

        new_data = {}   # 具体实例的信息
        for k,v in data.items():
            if isinstance(v, dict):
                new_data[k] = v

        if len(metric_split) == 1:
            new_metric = metric_split[0]
            if new_metric in data:       # 传统数据
                return data[new_metric]
            else:                           # 非pd
                new_metric_data = []
                for k, v in new_data.items():
                    if new_metric in v:
                        new_metric_data.append(v[new_metric])
                if len(new_metric_data) > 0:
                    return round(mean(new_metric_data), 6)
                else:
                    return None
                
        elif len(metric_split) == 2:          # pd
            inference_type, new_metric = metric_split
            new_metric_data = []
            for k, v in new_data.items():
                if "inference_type" in v and v["inference_type"] == inference_type and new_metric in v:
                    new_metric_data.append(v[new_metric])
            if len(new_metric_data) > 0:
                return round(mean(new_metric_data), 6)
            else:
                # 指标名含"-"的非pd数据
                for k, v in new_data.items():
                    if metric in v:
                        new_metric_data.append(v[new_metric])
                if len(new_metric_data) > 0:
                    return round(mean(new_metric_data), 6)
                else:
                    return None

    def get_instance_split_metric(self, data, metric):
        if self.get_instance_split_metric_base(data, metric) is not None:
            return self.get_instance_split_metric_base(data, metric)
        else:
            # 如果没有找到对应的指标，尝试将指标名中的"-"替换为"_"
            # new_metric = metric.replace('-', '_')
            # return self.get_instance_split_metric_base(data, new_metric)
            return None

    def translate_to_excel_according_metrics(self, metrics):
        data = {}
        for concurrency in self.concurrencies:
            for q in self.qps:
                # 构造以 concurrency 为行，latency 为列的 DataFrame
                data[(q, concurrency)] = analysis.results[str(concurrency)][str(q)]
        output_path = self.cache_file.replace('.json', '.xlsx')
        print(f'[translate_to_excel_according_metrics] output_path:{output_path}')
        with pd.ExcelWriter(output_path) as writer:
            for metric in metrics:
                for q in self.qps:
                    # 构造以 concurrency 为行，latency 为列的 DataFrame
                    data_new = [data[(q, concurrency)] for concurrency in self.concurrencies]
                    data_all = []
                    for single_data in data_new:
                        data_cur_concurrency = []
                        for label in self.labels:
                            if isinstance(metric, list):
                                for m in metric:
                                    data_cur_concurrency.append(self.get_instance_split_metric(single_data[label],m))
                                metric_num = len(metric)
                            elif isinstance(metric, str):
                                data_cur_concurrency.append(self.get_instance_split_metric(single_data[label],metric))
                                metric_num = 1
                        data_all.append(data_cur_concurrency)
                    df_qps = pd.DataFrame(data_all, index=self.concurrencies, columns=[item for item in self.labels for _ in range(metric_num)])
                    df_qps.index.name = 'concurrency'
                    sheet_name = f'qps_{q}_{metric}' if isinstance(metric, str) else f'qps_{q}_{",".join(metric)}'
                    # df_qps加一行写上指标名称
                    df_qps.loc['metric'] = [metric] * len(df_qps.columns)
                    df_qps.to_excel(writer, sheet_name=sheet_name)

class LogAnalysis_new(LogAnalysis):
    def __init__(self, model, qps, concurrencies, instance_deploy_msg, request_len=None, ):
        '''
        参数说明：
        model: 模型名称，如 'llama-7b' 或 'llama-13b'
        qps: 请求每秒数列表，如 [2, 4, 6, 8, 10, 12]
        concurrencies: 并发数列表，如 [1, 2, 4, 8, 16]
        instance_deploy_msg: 实例部署信息列表，如 [(prefill_tps, decode_tps)]，如 [([1,1,1,1],[]),([1,1],[2])]
        request_len: 请求长度('prompt_len-response_len'，如 '128-256')，默认为 None(表示采用ShareGPT数据集)
        '''
        self.path_tmp = 'A6000-2'
        self.model = model
        self.request_len = request_len
        self.distribution = 'poisson'
        self.concurrencies = concurrencies
        self.qps = qps
        self.instance_deploy_msg = instance_deploy_msg

        json_files = self.get_json_file_path(self.concurrencies[0], self.qps[0], 
                                             self.instance_deploy_msg, self.request_len)
        self.labels = json_files.keys()

        self.cache_file = f'results/results_cache-{self.path_tmp}-{model}-{self.request_len}-{self.qps}-{self.concurrencies}.json'
        print(f'[LogAnalysis] cache_file:{self.cache_file}')
        if os.path.exists(self.cache_file):
            print(f'[LogAnalysis] exist cache_file:{self.cache_file}')
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                self.results = json.load(f)
        else:
            self.results = {}
        
    def save_to_cache_file(self):
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

    def get_label(self, prefill_tps, decode_tps, is_pd=True):
        if not is_pd:
            res = ",".join(str(x) for x in prefill_tps)
        else:
            res = ",".join(str(x) for x in prefill_tps) + '-' + ",".join(str(x) for x in decode_tps)
        return res
    
    def get_json_file_path(self, concurrency, qps, instance_deploy_msg, request_len=None,):
        file_paths = {}
        if request_len is not None:
            req_msg = f'-{request_len}'
        else:
            req_msg = ''

        for prefill_tps, decode_tps in instance_deploy_msg:
            is_pd = len(decode_tps) != 0
            if not is_pd:
                # 非pd分离的json path
                instance_num = len(prefill_tps)
                tp = prefill_tps[0]
                json_file = f'/workspace/llm-serve/Llumnix/benchmark_test/logs/{self.path_tmp}-concurrency-{concurrency}{req_msg}/{self.model}/{self.distribution}/benchmark_{instance_num}_tp{tp}_2000_qps_{qps}_latency_info.json'
                file_paths[self.get_label(prefill_tps, None, False)] = json_file
            else:
                # pd分离的json path
                label = self.get_label(prefill_tps, decode_tps)
                json_file = f'/workspace/llm-serve/Llumnix/benchmark_test/logs/{self.path_tmp}-concurrency-{concurrency}{req_msg}/{self.model}/{self.distribution}/benchmark_pdd_2000_qps_{qps}_{label.replace("-","_")}_latency_info.json'
                file_paths[label] = json_file
        return file_paths

    def get_log_file_path(self, concurrency, qps, instance_deploy_msg, request_len=None):
        file_paths = {}
        if request_len is not None:
            req_msg = f'-{request_len}'
        else:
            req_msg = ''

        for prefill_tps, decode_tps in instance_deploy_msg:
            is_pd = len(decode_tps) != 0
            if not is_pd:
                # 非pd分离的log path
                instance_num = len(prefill_tps)
                tp = prefill_tps[0]
                log_file = f'/workspace/llm-serve/Llumnix/benchmark_test/logs/{self.path_tmp}-concurrency-{concurrency}{req_msg}/{self.model}/{self.distribution}/serve_{instance_num}_tp{tp}_2000_qps_{qps}.log'
                file_paths[self.get_label(prefill_tps, None, False)] = log_file
            else:
                # pd分离的log path
                label = self.get_label(prefill_tps, decode_tps)
                log_file = f'/workspace/llm-serve/Llumnix/benchmark_test/logs/{self.path_tmp}-concurrency-{concurrency}{req_msg}/{self.model}/{self.distribution}/serve_pdd_2000_qps_{qps}_{label.replace("-","_")}.log'
                file_paths[label] = log_file
        return file_paths
    
    def get_instance_file_path(self, concurrency, qps, instance_deploy_msg, request_len=None):
        file_paths = {}
        if request_len is not None:
            req_msg = f'-{request_len}'
        else:
            req_msg = ''

        for prefill_tps, decode_tps in instance_deploy_msg:
            is_pd = len(decode_tps) != 0
            if not is_pd:
                # 非pd分离的instance path
                instance_num = len(prefill_tps)
                tp = prefill_tps[0]
                instance_file = f'/workspace/llm-serve/Llumnix/benchmark_test/logs/{self.path_tmp}-concurrency-{concurrency}{req_msg}/{self.model}/{self.distribution}/serve_{instance_num}_tp{tp}_2000_qps_{qps}_instance.csv'
                file_paths[self.get_label(prefill_tps, None, False)] = instance_file
            else:
                # pd分离的instance path
                label = self.get_label(prefill_tps, decode_tps)
                instance_file = f'/workspace/llm-serve/Llumnix/benchmark_test/logs/{self.path_tmp}-concurrency-{concurrency}{req_msg}/{self.model}/{self.distribution}/serve_pdd_2000_qps_{qps}_{label.replace("-","_")}_instance.csv'
                file_paths[label] = instance_file
        return file_paths

    def get_lantency(self, json_file):
        '''
            return request_time, prefill_time, decode_time
        '''
        if not os.path.isfile(json_file):
            print(f"File {json_file} does not exist.")
            return None
        print(f'[get_lantency] Processing file: {json_file}')
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f'error:{str(e)}')
        assert len(data) == 1, "Expected data to contain only one entry"
        latencies = data[0]
        req_latencies, prefill_latencies, decode_latencies = latencies['request_latencies'], latencies['prefill_token_latencies'], latencies['decode_token_latencies']
        
        # per_token_latency_breakdown_list = data[0]['per_token_latency_breakdown_list']
        # prefill_waiting_time = [(per_token_latency_breakdown_list[i][0]['engine_step_timestamp_begin'] - per_token_latency_breakdown_list[i][0]['engine_add_request_timestamp'])*1000
        #                         for i in range(len(per_token_latency_breakdown_list))]
        # engine_step_latency_prefill = [per_token_latency_breakdown_list[i][0]['engine_step_latency'] for i in range(len(per_token_latency_breakdown_list))]
        # engine_step_latency_decode = [
        #     mean([token['engine_step_latency'] for token in per_token_latency_breakdown_list[i][1:]])
        #     if len(per_token_latency_breakdown_list[i][1:]) > 0 else None
        #     for i in range(len(per_token_latency_breakdown_list))
        # ]

        return {
            'request_time':round(mean(req_latencies), 4), 
            'prefill_time': round(mean(prefill_latencies), 4),
            'decode_time': round(mean(decode_latencies), 4),
            # 'prefill_waiting_time': round(mean(prefill_waiting_time), 4), 
            # 'prefill_step_time': round(mean(engine_step_latency_prefill), 4), 
            # 'decode_step_time': round(mean([x for x in engine_step_latency_decode if x is not None]), 4),
        }
    
    # Deprecated
    def get_step_lantency(self, json_file):
        '''
            return prefill_waiting_time, prefill_step_time, decode_step_time
        '''
        if not os.path.isfile(json_file):
            print(f"File {json_file} does not exist.")
            return None
        print(f'[get_step_lantency] Processing file: {json_file}')
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f'error:{str(e)}')
        assert len(data) == 1, "Expected data to contain only one entry"
        per_token_latency_breakdown_list = data[0]['per_token_latency_breakdown_list']
        prefill_waiting_time = [(per_token_latency_breakdown_list[i][0]['engine_step_timestamp_begin'] - per_token_latency_breakdown_list[i][0]['engine_add_request_timestamp'])*1000
                                for i in range(len(per_token_latency_breakdown_list))]
        engine_step_latency_prefill = [per_token_latency_breakdown_list[i][0]['engine_step_latency'] for i in range(len(per_token_latency_breakdown_list))]
        engine_step_latency_decode = [
            mean([token['engine_step_latency'] for token in per_token_latency_breakdown_list[i][1:]])
            if len(per_token_latency_breakdown_list[i][1:]) > 0 else None
            for i in range(len(per_token_latency_breakdown_list))
        ]

        return {
            'prefill_waiting_time': round(mean(prefill_waiting_time), 2), 
            'prefill_step_time': round(mean(engine_step_latency_prefill), 2), 
            'decode_step_time': round(mean([x for x in engine_step_latency_decode if x is not None]), 2)
        }
    

    def extract_migration_info(self, path, verbose=False):
        '''
        tp_hetero:不用
        res = {
            'avg_speed': avg_speed,
            'avg_migration_time': avg_migration_time,
            'avg_migrate_waiting_time': avg_migrate_waiting_time,
            'avg_migration_count': avg_migration_count,
            'avg_migration_aborted_dst_count': avg_migration_aborted_dst_count,
            'sum_migration_aborted_dst_count': avg_migration_aborted_dst_count*len(migration_info),
            'reject_migrate_out_count': reject_migrate_out_count,
            'reject_migrate_in_count': reject_migrate_in_count,
        }
        '''
        print(f"[extract_migration_info] Processing log file: {path}")

        # 检查文件是否存在
        if not os.path.isfile(path):
            print(f"File {path} does not exist.")
            return {}

        migration_info = {}
        reject_migrate_in_count = 0
        reject_migrate_out_count = 0
        # 示例： Instance ... migrate done, migrate request ['494c45676def4572986621d1afbc337f'], migration status: MigrationStatus.FINISHED, len: 7 blocks, cost: 240.65113067626953 ms
        # 正确的正则表达式应为：
        pattern = r"migrate request \[(.*?)\].*?len: (\d+) blocks,.*?cost: ([\d\.]+) ms"
        count = 0

        # 逐行读取文件（自动处理大文件）
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()

                if 'reject new migrate out' in line:
                    reject_migrate_out_count += 1
                if 'reject new migrate in' in line:
                    reject_migrate_in_count += 1
                # 获取迁移时间和速度
                if 'migrate done' in line and 'cost:' in line:
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
                                assert req_id in migration_info, f"{req_id},{type(req_id)},{line}"
                                migration_info[req_id]["blocks"] = blocks
                                migration_info[req_id]["time_ms"] = time
                                migration_info[req_id]["speed_blocks_per_s"] = speed

                if "engine_step_timestamp_end" in line or "_migrate_out_one_request start" in line \
                    or "MigrationStatus.ABORTED_DST, timestamps" in line \
                        or "MigrationStatus.ABORTED_SRC, timestamps" in line :
                    # 使用正则表达式提取请求 ID
                    request_id_match = re.search(r'[0-9a-f]{32}', line)
                    # 使用正则表达式提取时间戳
                    timestamp_match = re.search(r'timestamps: \d+\.\d+', line)

                    if request_id_match and timestamp_match:
                        request_id = request_id_match.group()
                        timestamp = float(timestamp_match.group().split(":")[1])

                        # 如果请求 ID 不在字典中，则初始化一个条目
                        if request_id not in migration_info:
                            migration_info[request_id] = {
                                "blocks": 0,
                                "time_ms":  0.0,
                                "speed_blocks_per_s": 0.0,
                                "engine_step_timestamp_end": None,
                                "migrate_out_one_request_start": None,
                                "migrate_start_count": 0,
                                "ABORTED_DST_count":0,
                                "ABORTED_SRC_count":0,
                            }

                        # 根据日志行内容更新对应的时间戳
                        if "engine_step_timestamp_end" in line:
                            migration_info[request_id]["engine_step_timestamp_end"] = timestamp
                        if migration_info[request_id]["time_ms"] == 0.0:
                            if "_migrate_out_one_request start" in line:
                                migration_info[request_id]["migrate_out_one_request_start"] = timestamp
                                migration_info[request_id]["migrate_start_count"] += 1
                            elif "MigrationStatus.ABORTED_DST, timestamps" in line:
                                migration_info[request_id]["ABORTED_DST_count"] += 1
                            elif "MigrationStatus.ABORTED_SRC, timestamps" in line:
                                migration_info[request_id]["ABORTED_SRC_count"] += 1
                                
                            if migration_info[request_id]["migrate_out_one_request_start"] is not None and migration_info[request_id]["engine_step_timestamp_end"] is not None:
                                migration_info[request_id]["migrate_waiting_time"] = (migration_info[request_id]["migrate_out_one_request_start"] - migration_info[request_id]["engine_step_timestamp_end"]) *1000
                                migration_info[request_id]["migrate_waiting_time"] = max(0, migration_info[request_id]["migrate_waiting_time"])
                                # migrate_waiting_times.append(migration_info[request_id]["migrate_waiting_time"])
                        else:
                            # print("not first migration")
                            pass

        fail_req_id = set()
        for req_id, info in migration_info.items():
            if 'migrate_waiting_time' not in migration_info[req_id]:
                fail_req_id.add(req_id)
                if verbose:
                    print(f'fail req_id:{req_id}, no migrate_waiting_time, {migration_info[req_id]}')
            else:
                if migration_info[req_id]["migrate_waiting_time"] > 1000:
                    pass
                    # file_output.write(f'req_id:{req_id},migrate_waiting_time:{migration_info[req_id]["migrate_waiting_time"]},ABORTED_DST_count:{migration_info[req_id]["ABORTED_DST_count"]}')
            if migration_info[req_id]['time_ms'] == 0.0:
                fail_req_id.add(req_id)
                if verbose:
                    print(f'fail req_id:{req_id}, no migrate_time, {migration_info[req_id]}')
        for req_id in fail_req_id:
            del migration_info[req_id]
        
        if migration_info:
            avg_speed = mean(info['speed_blocks_per_s'] for info in migration_info.values())
            avg_migration_time = mean(info['time_ms'] for info in migration_info.values())
            avg_migrate_waiting_time = mean(info["migrate_waiting_time"] for info in migration_info.values())
            avg_migration_count = mean(info['migrate_start_count'] for info in migration_info.values())
            avg_migration_aborted_dst_count = mean(info['ABORTED_DST_count'] for info in migration_info.values())
            if verbose:
                print(f"fail req num(lose msg): {len(fail_req_id)}, finished_flag:{finished_flag},finished_str:{finished_str}")
                print(f"Average migration speed: {avg_speed:.2f} blocks/s")
                print(f"Average migration time: {avg_migration_time:.2f} ms")
                print(f'Average migration waiting time: {avg_migrate_waiting_time:.2f} ms')
                print(f"Average migration count: {avg_migration_count}")
                print(f"Average migration ABORTED_DST count: {avg_migration_aborted_dst_count}")
                print(f"Sum migration ABORTED_DST count: {avg_migration_aborted_dst_count*len(migration_info)}")
                print(f"reject_migrate_out_count:{reject_migrate_out_count}")
                print(f"reject_migrate_in_count:{reject_migrate_in_count}")
                print(f"max block num : {max(info['blocks'] for info in migration_info.values())}")
            # if avg_migration_count > avg_migration_aborted_dst_count + 
            for req_id, info in migration_info.items():
                info['avg_speed_blocks_per_s'] = avg_speed
        else:
            print("No migration information found.")

        # assert finished_flag
        res = {
            'avg_speed': round(avg_speed,4),
            'avg_migration_time': round(avg_migration_time,4),
            'avg_migrate_waiting_time': round(avg_migrate_waiting_time,4),
            'avg_migration_count': round(avg_migration_count,4),
            'avg_migration_aborted_dst_count': round(avg_migration_aborted_dst_count,4),
            'sum_migration_aborted_dst_count': round(avg_migration_aborted_dst_count*len(migration_info),4),
            'reject_migrate_out_count': round(reject_migrate_out_count,4),
            'reject_migrate_in_count': round(reject_migrate_in_count,4),
        }
        return res
    
    def get_all_msg_qps(self, concurrency, qps):
        print(f"[get_all_msg_qps] concurrency:{concurrency}, qps:{qps}")
        json_files = self.get_json_file_path(concurrency, qps, 
                                            self.instance_deploy_msg, self.request_len,)
        log_files = self.get_log_file_path(concurrency, qps, 
                                            self.instance_deploy_msg, self.request_len,)
        instance_files = self.get_instance_file_path(concurrency, qps, 
                                            self.instance_deploy_msg, self.request_len,)
        labels = json_files.keys()
        res = {}
        for label in labels:
            is_pd = '-' in label
            latency = self.get_lantency(json_files[label])
            instance_ana = InstanceMetricsAnalysis(instance_files[label], is_pd)
            instance_ana.get_instance_metrics()
            if is_pd:
                migration_info = self.extract_migration_info(log_files[label],)
                res[label] = {**latency, **migration_info, **instance_ana.results}
            else:
                res[label] = {**latency, **instance_ana.results}
            print()
        return res
 
    def get_all_msg(self, cover=False):
        for concurrency in self.concurrencies:
            concurrency = str(concurrency)
            if concurrency not in self.results:
                self.results[concurrency] = {}
            for q in self.qps:
                q = str(q)
                if not cover and concurrency in self.results and q in self.results[concurrency]:
                    for label, results in self.results[concurrency][q].items():
                        if 'prefill_step_time' in results:
                            del results['prefill_step_time']
                        if 'decode_step_time' in results:
                            del results['decode_step_time']
                    continue
                results_qps = self.get_all_msg_qps(concurrency, q)
                self.results[concurrency][q] = results_qps
                self.save_to_cache_file()
        self.save_to_cache_file()

    def get_all_msg_qps_updata_instance_metric(self, concurrency, qps, data):
        print(f"[get_all_msg_qps_updata_instance_metric] concurrency:{concurrency}, qps:{qps}")
        instance_files = self.get_instance_file_path(concurrency, qps, 
                                            self.instance_deploy_msg, self.request_len,)
        labels = instance_files.keys()
        res = {}
        for label in labels:
            is_pd = '-' in label
            instance_ana = InstanceMetricsAnalysis(instance_files[label], is_pd)
            instance_ana.get_instance_metrics()
            data[label].update(instance_ana.results)
            print()
        return res
    
    def get_all_msg_updata_instance_metric(self):
        for concurrency in self.concurrencies:
            for q in self.qps:
                if str(concurrency) in self.results and str(q) in self.results[str(concurrency)]:
                    self.get_all_msg_qps_updata_instance_metric(concurrency, q, self.results[str(concurrency)][str(q)])
                    self.save_to_cache_file()

    def get_instance_split_metric_base(self, data, metric):

        metric_split = metric.split('-')

        new_data = {}   # 具体实例的信息
        for k,v in data.items():
            if isinstance(v, dict):
                new_data[k] = v

        if len(metric_split) == 1:
            new_metric = metric_split[0]
            if new_metric in data:       # 传统数据
                return data[new_metric]
            else:                           # 非pd
                new_metric_data = []
                for k, v in new_data.items():
                    if new_metric in v:
                        new_metric_data.append(v[new_metric])
                if len(new_metric_data) > 0:
                    return round(mean(new_metric_data), 6)
                else:
                    return None
                
        elif len(metric_split) == 2:          # pd
            inference_type, new_metric = metric_split
            new_metric_data = []
            for k, v in new_data.items():
                if "inference_type" in v and v["inference_type"] == inference_type and new_metric in v:
                    new_metric_data.append(v[new_metric])
            if len(new_metric_data) > 0:
                return round(mean(new_metric_data), 6)
            else:
                # 指标名含"-"的非pd数据
                for k, v in new_data.items():
                    if metric in v:
                        new_metric_data.append(v[new_metric])
                if len(new_metric_data) > 0:
                    return round(mean(new_metric_data), 6)
                else:
                    return None

    def get_instance_split_metric(self, data, metric):
        if self.get_instance_split_metric_base(data, metric) is not None:
            return self.get_instance_split_metric_base(data, metric)
        else:
            # 如果没有找到对应的指标，尝试将指标名中的"-"替换为"_"
            # new_metric = metric.replace('-', '_')
            # return self.get_instance_split_metric_base(data, new_metric)
            return None

    def translate_to_excel_according_metrics(self, metrics, suffix=None):
        data = {}
        for concurrency in self.concurrencies:
            for q in self.qps:
                # 构造以 concurrency 为行，latency 为列的 DataFrame
                data[(q, concurrency)] = self.results[str(concurrency)][str(q)]
        if suffix is not None:
            output_path = self.cache_file.replace('.json', f'_{suffix}.xlsx')
        else:
            output_path = self.cache_file.replace('.json', '.xlsx')
        print(f'[translate_to_excel_according_metrics] output_path:{output_path}')
        with pd.ExcelWriter(output_path) as writer:
            for metric in metrics:
                for q in self.qps:
                    # 构造以 concurrency 为行，latency 为列的 DataFrame
                    data_new = [data[(q, concurrency)] for concurrency in self.concurrencies]
                    data_all = []
                    for single_data in data_new:
                        data_cur_concurrency = []
                        for label in self.labels:
                            if isinstance(metric, list):
                                for m in metric:
                                    data_cur_concurrency.append(self.get_instance_split_metric(single_data[label],m))
                                metric_num = len(metric)
                            elif isinstance(metric, str):
                                data_cur_concurrency.append(self.get_instance_split_metric(single_data[label],metric))
                                metric_num = 1
                        data_all.append(data_cur_concurrency)
                    df_qps = pd.DataFrame(data_all, index=self.concurrencies, columns=[item for item in self.labels for _ in range(metric_num)])
                    df_qps.index.name = 'concurrency'
                    sheet_name = f'qps_{q}_{metric}' if isinstance(metric, str) else f'qps_{q}_{",".join(metric)}'
                    # df_qps加一行写上指标名称
                    df_qps.loc['metric'] = [metric] * len(df_qps.columns)
                    df_qps.to_excel(writer, sheet_name=sheet_name)

if __name__ == '__main__':
    # instance_metric = InstanceMetricsAnalysis('/workspace/llm-serve/Llumnix/benchmark_test/logs/A6000-2-formal2-concurrency-4-pdd-4/llama-13b/poisson/serve_pdd_tp1_2000_qps_1_1_3_instance.csv')
    # instance_metric.get_instance_metrics()
    # print(instance_metric.results)
    # print(analysis.get_all_msg_qps(1,1))

    # analysis = LogAnalysis('llama-7b')
    # analysis.get_all_msg()

    # analysis = LogAnalysis('llama-7b')
    # analysis.get_all_msg()

    # analysis = LogAnalysis('llama-7b')
    # analysis.get_all_msg_updata_instance_metric()

    # analysis = LogAnalysis('llama-13b')
    # analysis.get_all_msg_updata_instance_metric()

    # analysis = LogAnalysis('llama-13b')
    # metrics = [
    #         ["request_time", "prefill_time", "prefill-mofc",'prefill-gpu_cache_usage'],
    #         'prefill_bs',
    #         'prefill_all_time_bs',
    # ]
    # analysis.translate_to_excel_according_metrics(metrics)

    # instance_deploy_msg = [     # (prefill_tps, decode_tps)
    #     ([1,1,1,1],[]),([1,1],[2]),
    # ]
    # for req_len in ['512-256', '256-256', '128-256']:
    #     print(f'[main] prompt_len:{req_len}')
    #     analysis = LogAnalysis_new('llama-13b', [2], [1,2,4,8], instance_deploy_msg, request_len=req_len, )
    #     # analysis = LogAnalysis_new('llama-7b', [4], [1,2,4,8], instance_deploy_msg, request_len=req_len, )
    #     try:
    #         analysis.get_all_msg()
    #         metrics = [
    #                 ["request_time", "prefill_time", "decode_time", "prefill-mofc",'prefill-gpu_cache_usage'],
    #                 'prefill_bs',
    #                 'prefill_all_time_bs',
    #         ]
    #         analysis.translate_to_excel_according_metrics(metrics)
    #     except Exception as e:
    #         print(f'[main] error:{str(e)}')
            

    # 3卡
    instance_deploy_msg = [     # (prefill_tps, decode_tps)
        ([1,1,1],[]),([1],[2]),
    ]
    analysis = LogAnalysis_new('llama-13b', [1,2], [1,2,4], instance_deploy_msg, request_len=None, )
    analysis.get_all_msg()
    metrics = [
            ["request_time", "prefill_time", "decode_time", "prefill-mofc",'prefill-gpu_cache_usage'],
            'prefill_bs',
            'prefill_all_time_bs',
    ]
    analysis.translate_to_excel_according_metrics(metrics)
