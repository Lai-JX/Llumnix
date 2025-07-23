# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import threading
import time
from typing import List, Tuple, Optional, Callable

from llumnix.backends.utils import BarrierActor
import torch
from func_timeout import func_set_timeout, FunctionTimedOut
import ray
import ray.util.collective as col
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from vllm.worker.cache_engine import CacheEngine

from llumnix.internal_config import MigrationConfig
from llumnix.backends.migration_backend_interface import MigrationBackendBase
from llumnix.logging.logger import init_logger
from llumnix.constants import NUMPY_SUPPORTED_DTYPES_FOR_MIGRATION
from llumnix.utils import random_uuid
import numpy as np

logger = init_logger(__name__)

import subprocess
import threading
from queue import Queue

class MigrationBackendResource:
    """迁移后端资源"""
    def __init__(self, idx:int, buffer_shape:Tuple[int, int, int, int], buffer_dtype: str, buffer_device: str, local_rank: int):

        self.idx = idx
        self.dummy_cache = torch.empty(
            # size=(self.num_migration_buffer_blocks, self.num_layers, 2, self.migration_cache_size),
            size=buffer_shape,
            # dtype=self.cache_engine[0].dtype,
            dtype = buffer_dtype,
            # device=self.cache_device,
            device=buffer_device,
            pin_memory=True
        )
        self.local_rank = local_rank
        with torch.cuda.device(self.local_rank):
            self.migration_stream = torch.cuda.Stream()
        self.send_cache_split = None
        self.wait_for_split_event = threading.Event()

    def use(self):
        logger.info(f"正在使用迁移后端资源: {self.idx}")

    def __str__(self):
        return f"MigrationBackendResource(idx={self.idx})"


class MigrationBackendResourcPool:
    """管理一个迁移后端资源池，方便并发迁移"""
    def __init__(self, buffer_shape:Tuple[int, int, int, int], buffer_dtype: str, buffer_device: str, local_rank: int, pool_size:int = 10):
        """
        :param pool_size: 资源池大小
        """
        self.buffer_shape = buffer_shape
        self.buffer_dtype = buffer_dtype
        self.buffer_device = buffer_device
        self.local_rank =  local_rank

        self.pool_size = pool_size
        self.idx_backend = dict()           #   记录每个迁移后端的索引
        self.used_backend = set()               # 记录已使用的迁移后端(idx)
        self.backend_pool = Queue()                 # 用于存储所有迁移后端的idx
        self.lock = threading.Lock()
        self._initialize_pool()

    def _initialize_pool(self):
        """初始化资源池，预先创建所有 MappedPortResource 并放入队列"""
        with self.lock:
            for idx in range(self.pool_size):
                backend = MigrationBackendResource(idx, self.buffer_shape, self.buffer_dtype, self.buffer_device, self.local_rank)
                self.idx_backend [idx] = backend
                self.backend_pool.put(backend.idx)  # 将迁移后端的idx放入队列

    def acquire(self):
        """
        从资源池中获取一个MigrationBackendResource
        :return: MigrationBackendResource
        """
        with self.lock:
            if self.backend_pool.empty():
                raise RuntimeError("没有可用迁移后端了")

            backend_idx = self.backend_pool.get()
            while backend_idx in self.used_backend:  # 防止并发问题（理论上Queue已保证）
                if self.backend_pool.empty():
                    raise RuntimeError("没有可用迁移后端了")
                backend_idx = self.backend_pool.get()

            self.used_backend.add(backend_idx)
            logger.info(f"Get Migration Backend: {backend_idx}")
            return self.idx_backend[backend_idx]

    def release(self, backend_cache: MigrationBackendResource):
        """
        将MigrationBackendResource释放回资源池
        :param MigrationBackendResource
        """
        backend_idx = backend_cache.idx # 获取迁移后端的索引
        with self.lock:
            if backend_idx not in self.used_backend:
                raise ValueError(f"迁移后端 {backend_idx} 未被使用，无法释放")

            self.used_backend.remove(backend_idx)
            self.backend_pool.put(backend_idx)
            logger.info(f"Release Migration Backend: {backend_idx}")

    def get_available_count(self):
        """获取当前可用资源数量"""
        return self.backend_pool.qsize()



