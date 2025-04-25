import subprocess
import re
import os
import signal
import threading
import time
from typing import Optional, List
from collections import deque
import pynvml
from llumnix.logging.logger import init_logger

logger = init_logger(__name__)

class DCGMMonitor:
    def __init__(self, device_ids: List[int] = [0], interval_ms: int = 1000, max_output_lines: int = 100):
        """
        初始化 DCGM 监控器
        
        :param device_ids: GPU 设备 ID (如 "0")
        :param interval_ms: 采样间隔（毫秒）
        :param max_output_lines: 每个 Field ID 保存的最大输出行数
        """
        self.device_ids = device_ids
        self.interval_ms = interval_ms
        self.max_output_lines = max_output_lines
        self.device_ids_str = ",".join(map(str, self.device_ids))
        self.check_discovery()
        self.field_ids = ['100', '101'] + self._get_field_ids()

        self._process: Optional[subprocess.Popen] = None

        # 根据 max_output_lines 决定存储结构
        if max_output_lines <= 0:
            self._output_lines = []  # 不限制输出行数
            # 创建嵌套字典结构
            self.metrics_data = {}
            for field_id in self.field_ids:
                self.metrics_data[field_id] = {}
                for device_id in self.device_ids:
                    self.metrics_data[field_id][device_id] = []
        else:
            self._output_lines = deque(maxlen=max_output_lines)
            # 创建嵌套字典结构
            self.metrics_data = {}
            for field_id in self.field_ids:
                self.metrics_data[field_id] = {}
                for device_id in self.device_ids:
                    self.metrics_data[field_id][device_id] = deque(maxlen=max_output_lines)

        self._stop_thread = threading.Event()
        self._reader_thread: Optional[threading.Thread] = None
    
    def check_discovery(self) -> None:
        """执行 dcgmi discovery -l 命令以验证环境"""
        try:
            result = subprocess.run(
                ["dcgmi", "discovery", "-l"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10
            )
            logger.debug(f"[DEBUG] dcgmi discovery 输出:\n{result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"[WARNING] dcgmi discovery 命令执行失败: {e.stderr.strip()}")
            logger.info("[INFO] 尝试启动 nv-hostengine 并重试...")
            try:
                subprocess.run(
                    ["nv-hostengine"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=10
                )
                result = subprocess.run(
                    ["dcgmi", "discovery", "-l"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=10
                )
                logger.debug(f"[DEBUG] dcgmi discovery 输出（重试后）:\n{result.stdout.strip()}")
            except subprocess.CalledProcessError as retry_error:
                raise RuntimeError(f"dcgmi discovery 命令重试失败: {retry_error.stderr.strip()}")
            except Exception as retry_exception:
                raise RuntimeError(f"dcgmi discovery 检查重试失败: {str(retry_exception)}")
        except Exception as e:
            raise RuntimeError(f"dcgmi discovery 检查失败: {str(e)}")

    def _get_field_ids(self) -> List[str]:
        """提取 dcgmi profile 输出的 Field IDs"""
        try:
            result = subprocess.run(
                ["dcgmi", "profile", "-l", "-i", self.device_ids_str],
                capture_output=True,
                text=True,
                check=True,
                timeout=10
            )
            matches = re.findall(r'\|\s*(\d{4})\s*\|', result.stdout)
            field_ids = list(set(matches))  # 去重
            if len(field_ids) == 0:
                raise ValueError("未找到有效的 Field IDs")
            field_ids = sorted(field_ids, key=int)  # 按数值排序
            logger.debug(f"[DEBUG] 提取到 {len(field_ids)} 个监控指标")
            logger.info(f"[DEBUG] 提取到的 Field IDs: {field_ids}")
            return field_ids
        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] dcgmi 命令执行失败: {e.stderr.strip()}")
            return []
        except Exception as e:
            logger.error(f"[ERROR] 提取 Field IDs 失败: {str(e)}")
            return []

    def start(self):
        """启动 DCGM 监控"""
        try:
            cmd = [
                "stdbuf", "-oL",  # 强制标准输出为行缓冲模式
                "dcgmi", "dmon",
                "-i", self.device_ids_str,
                "-d", str(self.interval_ms),
                "-e", ",".join(self.field_ids)
            ]
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True
            )
            logger.info(f"[INFO] DCGM 监控已启动 | 设备: {self.device_ids} | PID: {self._process.pid}")

            self._stop_thread.clear()
            self._reader_thread = threading.Thread(target=self._read_output)
            self._reader_thread.start()
        except Exception as e:
            logger.error(f"[ERROR] 启动 DCGM 监控失败: {str(e)}")
            raise

    def _read_output(self):
        """定期读取后台进程的输出"""

        def deal_with_output(line: str):
            """处理输出行"""
            if not line.startswith("GPU"):
                return
            # 存储原始输出行
            self._output_lines.append(line.strip())
            logger.debug(f"[DEBUG] 实时输出: {line.strip()}")
            # 提取各指标
            metrics = line.strip().split()
            device_id, metrics = int(metrics[1]), metrics[2:]
            # print(device_id, metrics)
            for idx, metric in enumerate(metrics):
                self.metrics_data[self.field_ids[idx]][device_id].append(float(metric))

        try:
            wait_time = max(0.002, int(self.interval_ms) / 1000.0)
            while not self._stop_thread.is_set() and self._process.stdout:
                # 读取一行输出
                line = self._process.stdout.readline()
                deal_with_output(line)
                time.sleep(wait_time - 0.001)  # 根据 interval_ms 等待
        except Exception as e:
            logger.error(f"[ERROR] 读取输出失败: {str(e)}")

    def stop(self):
        """停止 DCGM 监控"""
        if self._process:
            try:
                self._stop_thread.set()
                if self._reader_thread and self._reader_thread.is_alive():
                    self._reader_thread.join()

                pgid = os.getpgid(self.pid)
                os.killpg(pgid, signal.SIGTERM)

                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(pgid, signal.SIGKILL)
                    self._process.wait()

                logger.info(f"[INFO] DCGM 监控已停止 | 设备: {self.device_ids}")
            except Exception as e:
                logger.error(f"[ERROR] 停止 DCGM 监控失败: {str(e)}")
            finally:
                self._process = None

    @property
    def is_running(self) -> bool:
        """检查监控是否在运行"""
        return self._process is not None and self._process.poll() is None
    @property
    def pid(self) -> Optional[int]:
        """获取监控进程 PID"""
        return self._process.pid if self._process else None
    
    def get_field_output(self, field_id: str) -> List[dict]:
        """
        获取指定 Field ID 的监控数据
        
        :param field_id: Field ID
        :return: 包含时间戳和值的列表
        """
        if field_id not in self._field_output_lines:
            raise ValueError(f"Field ID {field_id} 不存在")
        return list(self._field_output_lines[field_id])
    
    def get_metrics_data(self) -> dict:
        """
        获取所有指标数据
        
        :return: 
        """
        metrics_data = {}
        for field_id in self.field_ids:
            metrics_data[field_id] = {}
            for device_id in self.device_ids:
                metrics_data[field_id][device_id] = list(self.metrics_data[field_id][device_id])
        return metrics_data

    def get_all_output(self) -> List[str]:
        """
        获取所有原始输出行
        
        :return: 原始输出行的列表
        """
        return list(self._output_lines)

class PowerMonitor:
    def __init__(self, device_ids: List[int], interval_ms: int = 1000, max_output_lines: int = 10):
        """
        初始化 PowerMonitor
        
        :param device_ids: GPU 索引列表
        :param interval_ms: 采样间隔（毫秒）
        :param max_output_lines: 保存的最大输出行数，默认 10
        """
        self.device_ids = device_ids
        self.interval_ms = interval_ms / 1000.0  # 转换为秒
        self.max_output_lines = max_output_lines
        self.running = False
        self._stop_thread = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None

        if max_output_lines <= 0:
            self._output_lines = []  # 不限制输出行数
            self.metrics_data = {device_id : [] for device_id in self.device_ids}
        else:
            self.metrics_data = {device_id : deque(maxlen=max_output_lines) for device_id in self.device_ids}
            self._output_lines = deque(maxlen=max_output_lines)

        pynvml.nvmlInit()

    def start(self):
        """开始监控功耗"""
        self.running = True
        self._stop_thread.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_power)
        self._monitor_thread.start()
        logger.info(f"[INFO] Power 监控已启动 | 设备: {self.device_ids} | Thread Native ID: {threading.get_native_id()}")
    def _monitor_power(self):
        """监控功耗的线程"""
        try:
            while not self._stop_thread.is_set() :
                timestamp = time.time()
                power_readings = []
                for gpu_index in self.device_ids:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # 转换为瓦特
                    self.metrics_data[gpu_index].append(power)
                    power_readings.append(power)
                logger.debug(f"[DEBUG] 实时输出: {power_readings}")

                if self.max_output_lines > 0 and len(self._output_lines) >= self.max_output_lines:
                    self._output_lines.popleft()
                entry = {"timestamp": timestamp, "power": power_readings}
                self._output_lines.append(entry)

                logger.debug(f"[DEBUG] Power Data: {entry}")  # 打印当前功耗数据
                time.sleep(self.interval_ms)
        except Exception as e:
            logger.error(f"[ERROR] 监控功耗失败: {str(e)}")

    def get_power_data(self) -> dict:
        """
        获取功耗数据
        
        :return: 
        """
        power_data = {}
        for device_id in self.device_ids:
            power_data[device_id] = list(self.metrics_data[device_id])
        return power_data


    def stop(self):
        """停止监控"""
        self.running = False
        self._stop_thread.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join()
        pynvml.nvmlShutdown()
        logger.info(f"[INFO] Power 监控已停止 | 设备: {self.device_ids}")

    def get_output(self) -> List[dict]:
        """获取功耗监控数据"""
        return list(self._output_lines)

class GPUMonitor:
    def __init__(self, device_ids: List[int], interval_ms: int = 1000, max_output_lines: int = 10):
        """
        初始化 GPU 监控器
        
        :param device_ids: GPU 设备 ID (如 [0])
        :param interval_ms: 采样间隔（毫秒），默认 1000
        :param max_output_lines: 保存的最大输出行数，默认 10
        """
        self.device_ids = device_ids
        self.interval_ms = interval_ms
        self.max_output_lines = max_output_lines

        self.dcgm_monitor = DCGMMonitor(device_ids=device_ids, interval_ms=interval_ms, max_output_lines=max_output_lines)
        self.power_monitor = PowerMonitor(device_ids=device_ids, interval_ms=interval_ms, max_output_lines=max_output_lines)

    def start(self) -> bool:
        """
        启动指标监控
        
        :return: 是否成功启动
        """
        try:
            self.dcgm_monitor.start()  # 启动 DCGM 监控
            self.power_monitor.start()  # 启动 PowerMonitor
            return True
        except Exception as e:
            logger.error(f"[ERROR] 启动失败: {str(e)}")
            return False

    def stop(self) -> bool:
        """
        停止监控
        
        :return: 是否成功停止
        """
        try:
            self.dcgm_monitor.stop()  # 停止 DCGM 监控
            self.power_monitor.stop()  # 停止 PowerMonitor
            return True
        except Exception as e:
            logger.error(f"[ERROR] 停止失败: {str(e)}")
            return False
    def get_gpu_metrics(self) -> dict:
        """
        获取 GPU 监控数据
        
        :return: 包含dcgm profile和power数据的字典
        :rtype: dict
            {
                "gpu_metrics": {
                    "gpu_id": [value1, value2, ...],
                }
            }
        """
        power_data = self.power_monitor.get_power_data()
        gpu_metrics = self.dcgm_monitor.get_metrics_data()
        gpu_metrics['power'] = power_data
        return gpu_metrics

 