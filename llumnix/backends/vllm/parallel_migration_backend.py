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

import threading

import torch
import ray
from typing import Tuple, Any

from llumnix.logging.logger import init_logger
from llumnix.utils import ray_get_with_timeout
import numpy as np

logger = init_logger(__name__)

import subprocess
import threading
from queue import Queue


# Once worker died, proxy actor will not restart.
@ray.remote(num_cpus=0, max_concurrency=8, max_restarts=-1)
class ProxyActor:
    def __init__(self, is_driver_worker: bool, use_ray_spmd_worker: bool):
        self.is_driver_worker = is_driver_worker
        self.use_ray_spmd_worker = use_ray_spmd_worker

    def exec_method(self, handle: ray.actor.ActorHandle, from_driver_worker=None, *args, **kwargs) -> Any:
        if (from_driver_worker) is True or (from_driver_worker is None and self.is_driver_worker and not self.use_ray_spmd_worker):
            # logger.info(f"from_driver_worker:{from_driver_worker}, class name:{ray.get(handle.get_class_name.remote())}")
            ret = ray_get_with_timeout(
                handle.execute_engine_method_async.remote(
                    "execute_driver_worker_method_async", *args, **kwargs
                )
            )
        else:
            # logger.info(f"from_driver_worker:{from_driver_worker}, class name:{ray.get(handle.get_class_name.remote())}")
            ret = ray_get_with_timeout(
                handle.execute_method.options(concurrency_group="migate").remote(*args, **kwargs)
            )

        return ret

class MigrationBackendResource:
    """迁移后端资源"""
    def __init__(self, idx:int, buffer_shape:Tuple[int, int, int, int], buffer_dtype: str, buffer_device: str, local_rank: int):

        self.idx = idx
        pin_memory = (buffer_device == 'cpu')  # pin_memory only works for CPU tensors
        self.dummy_cache = torch.empty(
            # size=(self.num_migration_buffer_blocks, self.num_layers, 2, self.migration_cache_size),
            size=buffer_shape,
            # dtype=self.cache_engine[0].dtype,
            dtype = buffer_dtype,
            # device=self.cache_device,
            device=buffer_device,
            pin_memory=pin_memory
        )
        self.local_rank = local_rank
        if buffer_device == 'cpu':
            with torch.cuda.device(self.local_rank):
                self.migration_stream = torch.cuda.Stream()
        elif buffer_device.type == 'cuda':
            import cupy
            with cupy.cuda.Device(self.local_rank):
                self.migration_stream = cupy.cuda.Stream()
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
                logger.error(f"Already use backend: {self.used_backend}")
                raise RuntimeError("Without available migration backend resources")

            backend_idx = self.backend_pool.get()
            while backend_idx in self.used_backend:  # 防止并发问题（理论上Queue已保证）
                if self.backend_pool.empty():
                    raise RuntimeError("Without available migration backend resources")
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
    
    def require_according_idx(self, idx):
        """
        根据指定索引获取MigrationBackendResource
        :param idx: 指定的迁移后端索引
        :return: MigrationBackendResource
        """
        with self.lock:
            if idx not in self.idx_backend:
                raise ValueError(f"无效的迁移后端索引: {idx}")
            if idx in self.used_backend:
                raise RuntimeError(f"迁移后端 {idx} 当前已被使用")
            
            self.used_backend.add(idx)
            logger.info(f"根据索引获取Migration Backend: {idx}")
            return self.idx_backend[idx]

    def release_according_idx(self, idx):
        """
        根据指定索引释放MigrationBackendResource
        :param idx: 指定的迁移后端索引
        """
        with self.lock:
            if idx not in self.used_backend:
                raise ValueError(f"迁移后端 {idx} 未被使用，无法释放")
            
            self.used_backend.remove(idx)
            logger.info(f"根据索引释放Migration Backend: {idx}")

    def get_available_count(self):
        """获取当前可用资源数量"""
        return self.backend_pool.qsize()


class Group:
    """通信组"""
    def __init__(self, idx:int, group_name:str):

        self.idx = idx
        self.group_name_base = group_name
        self.group_name = group_name + f'_{self.idx}'

    def use(self):
        logger.info(f"正在使用通信组: {self.idx}")

    def __str__(self):
        return f"Group({self.group_name})"


class GroupPool:
    """管理一个通信组资源池，方便并发迁移"""
    def __init__(self, group_name, pool_size:int = 10):
        """
        :param pool_size: 资源池大小
        """

        self.pool_size = pool_size
        self.group_name = group_name

        self.idx_group = dict()           #   记录每个迁移后端的索引
        self.used_group = set()               # 记录已使用的迁移后端(idx)
        self.group_pool = Queue()                 # 用于存储所有迁移后端的idx
        self.lock = threading.Lock()
        self._initialize_pool()

    def _initialize_pool(self):
        """初始化通信组，预先创建所有 Group 并放入队列"""
        with self.lock:
            for idx in range(self.pool_size):
                group = Group(idx, self.group_name)
                self.idx_group [idx] = group
                self.group_pool.put(group.idx)  # 将迁移后端的idx放入队列

    def acquire(self):
        """
        从资源池中获取一个Group
        :return: Group
        """
        with self.lock:
            if self.group_pool.empty():
                logger.error(f"Already use Group: {self.used_group}")
                raise RuntimeError("Without available migration backend resources")

            group_idx = self.group_pool.get()
            while group_idx in self.used_group:  # 防止并发问题（理论上Queue已保证）
                if self.group_pool.empty():
                    raise RuntimeError("Without available migration backend resources")
                group_idx = self.group_pool.get()

            self.used_group.add(group_idx)
            logger.info(f"Get Communication Group: {group_idx}")
            return self.idx_group[group_idx]

    def release(self, group: Group):
        """
        将MigrationBackendResource释放回资源池
        :param MigrationBackendResource
        """
        group_idx = group.idx # 获取迁移后端的索引
        with self.lock:
            if group_idx not in self.used_group:
                raise ValueError(f"Group {group_idx} 未被使用，无法释放")

            self.used_group.remove(group_idx)
            self.group_pool.put(group_idx)
            logger.info(f"Release Communication Group: {group_idx}")

    def get_available_count(self):
        """获取当前可用资源数量"""
        return self.group_pool.qsize()
